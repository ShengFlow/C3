/**
 * @file C3BackwardCapture.h
 * @generation SHARED 跨代共有（JIT-2.0/2.x/3.0 共用后端捕获）
 * @brief 反向图 JIT 捕获与编译引擎
 * @details 在 autograd 的 backward 执行路径中插入 C3 编译/执行尝试。
 *          核心流程：forward 时同步记录 C3 Graph 节点 → backward 时查 C3KernelRegistry
 *          → 命中则执行编译后的 backward kernel → 未命中则回退 eager + 异步编译。
 *
 *          支持的反向映射表：
 *          - ReLUNode: Mul(Gt(x, 0), grad) → 新增 GtNode
 *          - SigmoidNode: Mul(Mul(Sigmoid(x), Sub(1, Sigmoid(x))), grad) → 已有节点
 *          - TanhNode: Mul(Sub(1, Mul(Tanh(x), Tanh(x))), grad) → 已有节点
 *          - AddNode: grad, grad（广播时 SumReduce）→ 新增 SumReduceNode
 *          - MulNode: Mul(grad, B), Mul(A, grad) → 已有节点
 *          - MatMulNode: MatMul(grad, Transpose(B)), MatMul(Transpose(A), grad) → 新增 TransposeNode
 *          - NegNode: Neg(grad) → 已有节点
 *          - SubNode: grad, Neg(grad) → 已有节点
 *          - DivNode: Div(grad, B), Neg(Mul(A, Div(grad, Mul(B, B)))) → 已有节点
 *          - ExpNode: Mul(Exp(x), grad) → 新增 ExpNode
 *          - LogNode: Div(grad, x) → 已有节点
 *
 *          多输出策略（MLIR 后端函数签名仅支持单输出指针）：
 *          多输入节点（Add/Mul/MatMul/Sub/Div）的 backward 需产生多个梯度，
 *          这里不改造 codegen，而是为每个上游梯度编译一个独立单输出 kernel，
 *          注册 key 追加 "|in:<i>" 后缀区分。tryExecuteBackward 遍历输入逐 key
 *          查找并执行，任一缺失则整体回退 eager 以保证正确性。
 *
 *          回退策略：
 *          - 形状不匹配 → 静默回退到 eager（预期行为）
 *          - 未注册节点类型 → 静默回退到 eager
 *          - 任一输入梯度缺少编译 kernel → 整体回退到 eager
 *          - 执行异常 → 回退 + 记录日志（不影响训练正确性）
 * @date 2026/8/4
 */

#ifndef CTORCH_C3_BACKWARD_CAPTURE_H
#define CTORCH_C3_BACKWARD_CAPTURE_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "C3Engine.h"
#include "C3KernelRegistry.h"
#include "Graph.h"
#include "AutoGrad/Node.h"

namespace ct {
namespace c3 {

/**
 * @class C3BackwardCapture
 * @brief 反向图 JIT 捕获与编译引擎
 * @details 单例类，负责将 autograd 节点的 backward 捕获为 C3 Graph 并编译。
 *          与 C3KernelRegistry 集成以实现热替换。
 */
class C3BackwardCapture {
public:
    /**
     * @brief backward 子图构建结果：{Graph, fwd_input_map}
     * @details fwd_input_map[k] = 该反向图的第 (k+1) 个输入（grad 之后的第 k 个）
     *          对应的 forward_inputs 索引。因最小集 build 只加实际用到的 forward 输入，
     *          图输入顺序 ≠ forward_inputs 顺序，运行时必须按此表喂入（见 C3KernelRegistry）。
     */
    using BackwardGraph = std::pair<Graph, std::vector<size_t>>;

    /** @brief 获取 C3BackwardCapture 单例实例 */
    static C3BackwardCapture& getInstance();

    // 后台 detached 编译任务的生命周期计数，仅供异步提交点和清理护栏使用。
    bool taskStarted();
    void taskFinished();

    /**
     * @brief 等待所有后台反向编译任务结束。
     * @details 后台任务使用 detached thread 以避免 std::future 析构阻塞；
     *          调用本方法可在 C3/LLVM 静态资源释放前安全回收这些任务。
     */
    void shutdown();

    /**
     * @brief 清除反向融合捕获器中的所有临时状态与缓存。
     * @details 消除测试用例间的状态和残留节点的交叉污染，确保完整的环境隔离。
     */
    void clear();

    /**
     * @brief [优化 2026-08-16] 清理单次 backward 调用范围内的临时状态。
     * @details 在每次反向传播结束时调用，清空未被消费的截获梯度与 miss marker 节点，
     *          防止因为内存地址复用导致 Stale Tensor / UAF 崩溃或错误的梯度匹配。
     */
    void clearCallScopedState();

    /**
     * @brief 尝试执行编译后的 backward kernel
     * @param node 当前 autograd 节点
     * @param grad 下游梯度张量
     * @return 若命中 C3 缓存且执行成功，返回上游梯度列表；否则返回 nullopt
     * @details 首先查 C3KernelRegistry 是否有匹配的 backward kernel。
     *          命中则执行，未命中则返回 nullopt（调用方回退 eager）。
     *          执行失败时静默返回 nullopt，不抛异常。
     */
    std::optional<std::vector<Tensor>> tryExecuteBackward(
        const ::Node* node, const Tensor& grad,
        const std::vector<Tensor>& forward_inputs = {});

    /**
     * @brief 异步编译 backward 子图
     * @param node 当前 autograd 节点
     * @param grad 下游梯度张量
     * @details 在后台线程中构建 C3 Graph 并编译，编译完成后自动注册到 C3KernelRegistry。
     *          若同一节点类型 + 形状的编译任务已在进行中，去重。
     *          编译失败时静默处理，不影响训练流程。
     */
    void compileBackwardAsync(const ::Node* node, const Tensor& grad);

    /**
     * @brief MIMO 统一反向融合执行尝试
     */
    std::optional<std::vector<Tensor>> tryExecuteUnifiedMIMOBackward(
        const ::Node* node, const Tensor& grad,
        const std::vector<Tensor>& forward_inputs);

    /**
     * @brief MIMO 统一反向融合异步编译
     */
    void compileUnifiedMIMOBackwardAsync(
        const ::Node* relu_node, const ::Node* add_node, const ::Node* matmul_node,
        const TensorDesc& grad_desc, const TensorDesc& z_desc,
        const TensorDesc& x_desc, const TensorDesc& w_desc);

    /**
     * @brief 无 bias SwiGLU FFN 反向 MIMO 异步编译 (2026-09-06)
     * @details out = h @ W_d, h = g*u, g = silu(gate_pre), gate_pre = x @ W_g, u = x @ W_u。
     *          一次 kernel 算 9 个梯度输出(grad_h/grad_W_d/grad_g/grad_u/grad_gate_pre/
     *          grad_x_gate/grad_W_g/grad_x_up/grad_W_u), pending 表回填 Mul/SiLU/两个 MatMul。
     */
    void compileFFNMIMOBackwardAsync(
        const ::Node* mm_out_node, const ::Node* mul_node, const ::Node* silu_node,
        const ::Node* gate_mm, const ::Node* up_mm,
        const TensorDesc& grad_desc, const TensorDesc& x_desc, const TensorDesc& wg_desc,
        const TensorDesc& wu_desc, const TensorDesc& wd_desc, const TensorDesc& g_desc,
        const TensorDesc& u_desc, const TensorDesc& h_desc, const TensorDesc& gp_desc);

    /**
     * @brief 为指定输入索引异步编译 backward 单输出 kernel
     * @param node 当前 autograd 节点
     * @param grad 下游梯度张量
     * @param input_index 目标上游输入索引
     * @details tryExecuteBackward 对多输入节点逐 key 查找时，若某输入缺失，
     *          仅触发该输入的编译，其余已编译输入保持缓存复用。
     */
    void compileBackwardAsyncForInput(const ::Node* node, const Tensor& grad,
                                      size_t input_index);

    /**
     * @brief 构建反向子图的 C3 Graph
     * @param node 当前 autograd 节点
     * @param grad_desc 下游梯度的 TensorDesc
     * @param input_descs forward 输入的 TensorDesc 列表
     * @return 构建好的 C3 Graph；若无法构建（节点类型不支持），返回 nullopt
     * @details 根据 node 的类型，选择对应的反向规则，构建等价的 C3 Graph。
     *          例如 ReLUNode 构建 Mul(Gt(x, 0), grad) 的图。
     */
    std::optional<Graph> buildBackwardGraph(
        const ::Node* node,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    /**
     * @brief 为指定输入索引构建反向子图的单输出 C3 Graph
     * @param node 当前 autograd 节点
     * @param input_index 目标上游输入索引（多输入节点 0..N-1；单输入节点必须为 0）
     * @param grad_desc 下游梯度的 TensorDesc
     * @param input_descs forward 输入的 TensorDesc 列表
     * @return 单输出 C3 Graph（仅计算目标输入的梯度）；不支持则返回 nullopt
     * @details 多输入节点（Add/Mul/MatMul/Sub/Div）的 backward 需产生多个梯度，
     *          每个梯度编译为独立单输出 kernel，用 input_index 区分。
     *          单输入节点（ReLU/Sigmoid/Tanh/Neg/Exp/Log）仅支持 input_index == 0。
     */
    std::optional<Graph> buildBackwardGraphForInput(
        const ::Node* node,
        size_t input_index,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    /**
     * @brief 按节点类型字符串 + 输入索引构建反向子图的单输出 Graph
     * @param node_type 节点类型字符串（如 "ReLUNode"、"AddNode"）
     * @param input_index 目标上游输入索引
     * @param grad_desc 下游梯度的 TensorDesc
     * @param input_descs forward 输入的 TensorDesc 列表
     * @return 单输出 C3 Graph；不支持则返回 nullopt
     * @details 供异步编译线程使用（不持有 Node 指针，规避 UAF），
     *          与 buildBackwardGraphForInput 逻辑一致，仅改用字符串分发。
     */
    std::optional<BackwardGraph> buildBackwardGraphForTypeAndIndex(
        const std::string& node_type,
        size_t input_index,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    /**
     * @brief 检查节点类型是否支持 C3 backward 编译
     * @param node_type 节点类型字符串
     * @return true 如果支持
     */
    static bool supportsNodeType(const std::string& node_type);

    /**
     * @brief 等待所有 in-flight 反向编译任务完成（轮询实现，简单可靠）
     * @details [P0.6B 2026-08-30 苏璃珞 重做] miss 后等所有 pending_compiles_
     *          任务 erase 完才返回。**主线程阻塞** 但保证之后同 key 必命中。
     *          实测：test_c3_backward 6 ReLU + 6 Sigmoid 全跑完 < 1s。
     *          替代方案：CV 实现（需要新 condition_variable）—— 留给后续。
     */
    void waitForPendingCompiles() {
        while (true) {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            if (pending_compiles_.empty()) return;
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /** @brief 获取编译统计信息 */
    struct Stats {
        size_t capture_count = 0;      ///< 捕获次数
        size_t compile_count = 0;      ///< 编译次数
        size_t cache_hit_count = 0;    ///< 缓存命中次数
        size_t cache_miss_count = 0;   ///< 缓存未命中次数
        size_t execution_failures = 0; ///< 执行失败次数
        size_t fusion_compile_count = 0; ///< 融合编译次数
        size_t fusion_hit_count = 0;    ///< 融合执行命中次数
        size_t fusion_miss_count = 0;   ///< 融合尝试未命中次数
        size_t mimo_compile_count = 0;  ///< MIMO(ReLU/Add/MatMul 反向融合)编译次数
        size_t mimo_hit_count = 0;      ///< MIMO 执行命中次数
        size_t mimo_miss_count = 0;     ///< MIMO 尝试未命中(无已编译 kernel)次数
        uint64_t mimo_exec_us = 0;      ///< MIMO kernel->execute() 累计耗时(us)
        uint64_t mimo_keybuild_us = 0;  ///< MIMO cache key 字符串构建累计耗时(us)

        // ========== [P0.1 2026-08-30 苏璃珞] backward fallback 覆盖率统计 ==========
        // 目的：量化 C3 backward 真实覆盖率（C3 命中 vs eager fallback），给 P0/P1 提供数据基线
        // 之前洛锦的批判性评估指出："C3 backward 不是完整自动微分后端"，但**没有量化数据**。
        // 这两个字段让"覆盖率"可测量、可追踪、可报告。
        size_t backward_attempt_count = 0;            ///< tryExecuteBackward 总调用次数
        size_t backward_c3_attempt_count = 0;        ///< 走 C3 路径（compile + execute kernel）
        size_t backward_eager_fallback_count = 0;    ///< fallback 到 eager 的次数
        /// fallback 原因分类（unsupported_node_type / kernel_not_found / shape_mismatch / ...）
        /// key = 原因字符串，value = 出现次数
        std::unordered_map<std::string, size_t> backward_fallback_reasons;
        /// backward fallback 覆盖率 = 1 - eager_fallback / attempt
        /// 计算：getStats() 调用方算，Stats 不缓存派生指标
    };

    Stats getStats() const;

    // ======================= 反向融合检测 (Phase 2) =======================

    /**
     * @brief 记录一个 backward 节点（用于序列检测）
     * @param node_type 节点类型字符串
     * @param grad_shape 下游梯度形状
     * @param input_shape forward 输入形状
     * @details 在 ComputeCore::backward 中每次处理完一个节点后调用。
     *          当连续多个 backward 节点形成 fusion 模式时，触发融合编译。
     */
    void recordBackwardNode(const std::string& node_type,
                             const std::vector<size_t>& grad_shape,
                             const std::vector<size_t>& input_shape,
                             const std::vector<Tensor>& forward_inputs);

    /**
     * @brief 尝试执行已编译好的反向融合 kernel
     * @param node 当前 autograd 节点（用于拿到 forward_inputs）
     * @param grad 最下游梯度张量
     * @param forward_inputs forward 阶段输入张量列表
     * @return 若命中反向融合缓存，返回最终梯度（单张量，对应最上游的目标输入）；否则返回 nullopt
     * @details 优先于逐输入单 kernel 执行。
     *          融合 kernel 对应一段连续的 backward 序列（如 ReLU → Sigmoid → Mul），
     *          命中后一次性跑完整个序列，节省中间写读。
     */
    std::optional<Tensor> tryExecuteFusedBackward(
        const ::Node* node,
        const Tensor& grad,
        const std::vector<Tensor>& forward_inputs);

private:
    C3BackwardCapture() = default;
    ~C3BackwardCapture() { shutdown(); }
    C3BackwardCapture(const C3BackwardCapture&) = delete;
    C3BackwardCapture& operator=(const C3BackwardCapture&) = delete;

    // ======================= 反向 Graph 构建助手 =======================

    /**
     * @brief 构建 ReLU backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(Gt(x, 0), grad)
     */
    BackwardGraph buildReLUBackwardGraph(const TensorDesc& grad_desc,
                                  const TensorDesc& input_desc);

    /**
     * @brief 构建 Sigmoid backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(Mul(Sigmoid(x), Sub(1, Sigmoid(x))), grad)
     */
    BackwardGraph buildSigmoidBackwardGraph(const TensorDesc& grad_desc,
                                     const TensorDesc& input_desc);

    /**
     * @brief 构建 SiLU backward 的 C3 Graph (PEL25 #10)
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(grad, Add(Sigmoid(x), Mul(x, Mul(Sigmoid(x), Sub(1, Sigmoid(x))))))
     *   即: d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     *        grad_input = grad * d_silu_dx
     */
    BackwardGraph buildSiLUBackwardGraph(const TensorDesc& grad_desc,
                                   const TensorDesc& input_desc);

    /**
     * @brief 构建 SwiGLU backward 的 C3 Graph (PEL25 #10, 双输入)
     * @param grad_desc 下游梯度描述符
     * @param input_descs forward 输入描述符 [x, gate]
     * @param input_index 目标上游输入索引 (0=x → dL/dx, 1=gate → dL/dgate)
     * @return 单输出 C3 Graph (仅计算目标输入的梯度)
     *   input_index=0: grad_x = grad * gate * silu_d(x)
     *   input_index=1: grad_gate = grad * silu(x)
     */
    BackwardGraph buildSwiGLUBackwardGraph(const TensorDesc& grad_desc,
                                    const std::vector<TensorDesc>& input_descs,
                                    size_t input_index);

    /**
     * @brief 构建 Tanh backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(Sub(1, Mul(Tanh(x), Tanh(x))), grad)
     */
    BackwardGraph buildTanhBackwardGraph(const TensorDesc& grad_desc,
                                  const TensorDesc& input_desc);

    /**
     * @brief 构建 Add backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param lhs_desc forward 左输入描述符
     * @param rhs_desc forward 右输入描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: grad（广播时 SumReduce 缩小）
     */
    BackwardGraph buildAddBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& lhs_desc,
                                 const TensorDesc& rhs_desc,
                                 size_t input_index);

    /**
     * @brief 构建 Mul backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param a_desc forward 左输入描述符
     * @param b_desc forward 右输入描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: Mul(grad, B) 或 Mul(A, grad)
     */
    BackwardGraph buildMulBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 size_t input_index);

    /**
     * @brief 构建 MatMul backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param a_desc forward 左输入描述符
     * @param b_desc forward 右输入描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: MatMul(grad, Transpose(B)) 或 MatMul(Transpose(A), grad)
     */
    BackwardGraph buildMatMulBackwardGraph(const TensorDesc& grad_desc,
                                    const TensorDesc& a_desc,
                                    const TensorDesc& b_desc,
                                    size_t input_index);

    /**
     * @brief 构建 Neg backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @return C3 Graph: Neg(grad)
     */
    BackwardGraph buildNegBackwardGraph(const TensorDesc& grad_desc);

    /**
     * @brief 构建 Sub backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: grad 或 Neg(grad)
     */
    BackwardGraph buildSubBackwardGraph(const TensorDesc& grad_desc,
                                 size_t input_index);

    /**
     * @brief 构建 Div backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param a_desc forward 左输入（被除数）描述符
     * @param b_desc forward 右输入（除数）描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: Div(grad, B) 或 Neg(Mul(A, Div(grad, Mul(B, B))))
     */
    BackwardGraph buildDivBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 size_t input_index);

    /**
     * @brief 构建 Exp backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @param output_desc forward 输出描述符（exp(x) 的值）
     * @return C3 Graph: Mul(Exp(x), grad) = Mul(output, grad)
     */
    BackwardGraph buildExpBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& input_desc,
                                 const TensorDesc& output_desc);

    /**
     * @brief 构建 Log backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Div(grad, x)
     */
    BackwardGraph buildLogBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& input_desc);

    /**
     * @brief Softmax 反向 Graph（单输入节点：dim 是 attribute，axis 固定 1）
     * @param grad_desc 下游梯度描述符（与 y 同形）
     * @param input_desc forward 输入描述符
     * @return C3 Graph: y * (grad - sum(grad*y, dim=1, keepdim))
     * @details 7 op 分解：
     *          1) 重算 y = softmax(x)  （exp + sum_reduce[keepdim] + div）
     *          2) grad * y            （mul）
     *          3) sum_reduce[keepdim]  （axis=1, keepdim=true → [M, 1]）
     *          4) grad - sum          （sub, 广播 [M,1] → [M,N]）
     *          5) y * diff            （mul）
     */
    BackwardGraph buildSoftmaxBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& input_desc);

    /**
     * @brief CrossEntropy 反向 Graph（双输入节点：logits + target 都是 [M, N]）
     * @param grad_desc 下游梯度描述符（CE 反向通常为常数 1/M 或 1，公式不依赖）
     * @param input_descs_0 logits 描述符
     * @param input_descs_1 target 描述符（one-hot / soft probability，[M, N]）
     * @return C3 Graph: softmax(logits) - target  （4 op：Exp + SumReduce[keepdim] + Div + Sub）
     * @details target 不需要 grad，input_index=1 应返回 std::nullopt。
     *          仅 input_index=0（logits 的梯度）有意义。
     */
    BackwardGraph buildCrossEntropyBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& input_descs_0,
                                 const TensorDesc& input_descs_1);

    /**
     * @brief 根据节点类型字符串构建 backward Graph（用于融合编译）
     * @param node_type 节点类型字符串（如 "ReLUNode", "AddNode"）
     * @param grad_desc 下游梯度描述符
     * @param input_descs forward 输入描述符列表
     * @return 构建好的 C3 Graph；若类型不支持，返回 nullopt
     * @details 与 buildBackwardGraph 功能相同，但使用字符串匹配而非 typeid。
     *          用于融合编译场景，此时只有节点类型字符串，无实际节点对象。
     */
    std::optional<BackwardGraph> buildBackwardGraphForType(
        const std::string& node_type,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    // ======================= 工具函数 =======================

    /**
     * @brief 判断是否需要 SumReduce（用于广播反向）
     * @param grad_shape 梯度的形状（当前接收到的）
     * @param target_shape 上游节点期望的形状
     * @return true 如果 grad_shape 需要 SumReduce 才能匹配 target_shape
     */
    static bool needsSumReduce(const std::vector<size_t>& grad_shape,
                                const std::vector<size_t>& target_shape);

    /**
     * @brief 构建 SumReduce 的 axis 参数
     * @param grad_shape 当前梯度形状
     * @param target_shape 目标形状
     * @return 需要 reduce 的 axis（从 grad_shape 到 target_shape）
     */
    static int computeReduceAxis(const std::vector<size_t>& grad_shape,
                                  const std::vector<size_t>& target_shape);

    // ======================= 统计信息 =======================

    mutable std::mutex stats_mutex_;
    size_t capture_count_ = 0;
    size_t compile_count_ = 0;
    size_t cache_hit_count_ = 0;
    size_t cache_miss_count_ = 0;
    size_t execution_failures_ = 0;
    size_t fusion_compile_count_ = 0;

    // 去重 map：正在编译中的 (node_type + shape_hash)
    std::mutex pending_mutex_;
    std::unordered_map<std::string, bool> pending_compiles_;

    // detached 后台编译任务的生命周期护栏：shutdown() 等待 active_tasks_ 归零，
    // 防止任务在单例/LLVM 资源析构后继续访问 this。
    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::atomic<size_t> active_tasks_{0};
    std::atomic<bool> shutting_down_{false};

    // ======================= 反向融合检测 (Phase 2) =======================

    static constexpr size_t kFusionWindowSize = 4; ///< 融合检测窗口大小
    static constexpr int kFusionThreshold = 1;     ///< 触发热融合的频次阈值（=1 表示第一次出现就异步编译，降低用户冷启动首跳延迟）

    /// 反向序列条目
    struct BackwardSequence {
        std::vector<std::string> node_types;         ///< 节点类型序列（[0]=最下游, [N-1]=最上游, 反向传播执行顺序与 ComputeCore 一致）
        std::vector<std::vector<size_t>> grad_shapes; ///< 与 node_types 对齐：每个节点收到的下游 grad 形状（grad_shapes[0] 为最下游端 dL/dy 形状）
        std::vector<std::vector<size_t>> input_shapes;///< 与 node_types 对齐：每个节点的首个 forward 输入形状（input_shapes.back() 为最上游端 data 形状）
        int frequency = 0;                           ///< 出现频次
        bool compiling = false;                      ///< 是否正在编译
    };

    /// 最近的 backward 节点序列（RingBuffer）
    std::deque<std::string> recent_sequence_;
    std::deque<std::vector<size_t>> recent_grad_shapes_;  ///< 与 recent_sequence_ 对齐：每个节点的下游 grad 形状
    std::deque<std::vector<size_t>> recent_input_shapes_; ///< 与 recent_sequence_ 对齐：每个节点的首个 forward 输入形状
    std::deque<std::vector<Tensor>> recent_forward_inputs_; ///< 与 recent_sequence_ 对齐：每个节点的完整 forward inputs（执行融合时按序取）

    /**
     * @brief 融合拦截的待取结果：N0 执行融合时一次性算出 outs[w]=w 个节点的 upstream grad，
     *        outs[1..w-1] 存到这里；当 N1..Nw-1 依次进入 tryExecuteBackward 时直接取出返回，
     *        避免重复计算 & 严格对齐 ComputeCore 的 grad-pack 分发流程。
     * value = {节点类型名, 对应 upstream grad Tensor}：type 校验防止 raw ptr 地址复用的误命中。
     */
    std::unordered_map<const ::Node*, std::pair<std::string, Tensor>> pending_intercepted_;
    std::unordered_map<const ::Node*, std::vector<Tensor>> pending_mimo_intercepted_;
    mutable std::shared_mutex intercepted_mutex_; ///< 读写锁：miss 路径只读拿共享锁（大幅降低开销），写入 pending 时才拿独占锁

    /**
     * @brief 本轮 backward 中已确认 "fusion lookup 失败" 的节点指针标记。
     *        在同一个 backward 轮次中，一个节点若已经走过路径 B 的 upstream traversal 且 lookup 失败，
     *        则后续再次被访问（例如 wrapper 节点转发）时直接跳过 B 路径，避免重复图遍历开销。
     */
    std::unordered_set<const ::Node*> miss_marker_nodes_;
    mutable std::mutex miss_marker_mutex_;

    /// 已观察到的序列及其频次
    std::unordered_map<std::string, BackwardSequence> sequence_counts_;

    /// 融合检测 mutex
    std::mutex fusion_mutex_;

    /**
     * @brief 检查节点类型是否为元素操作（可融合）
     */
    static bool isElementWiseBackward(const std::string& node_type);

    /**
     * @brief 构建序列 key
     */
    static std::string makeSequenceKey(const std::vector<std::string>& types);

    /**
     * @brief 检查序列是否可融合
     * @param types 节点类型序列
     * @return true 如果序列中的所有节点都是元素操作且可融合
     */
    static bool isFusableSequence(const std::vector<std::string>& types);

    /**
     * @brief 为融合序列构建 fused backward Graph 并异步编译
     */
    void compileFusedBackwardAsync(const BackwardSequence& seq);

    // ======================= 融合查找辅助 =======================

    /**
     * @brief 构造带形状签名的反向融合注册/查找 key（注册与查找共用，保证格式对齐）
     * @param seq_key 由 makeSequenceKey 得到的短序列 key（如 "ReLU+Sigmoid"）
     * @param grad_shape 下游梯度形状
     * @param input_shape 首个 forward 输入形状
     */
    static std::string makeFusedBackwardKey(const std::string& seq_key,
                                             const std::vector<size_t>& grad_shape,
                                             const std::vector<size_t>& input_shape);

    /**
     * @brief 从 recent_sequence_ 尾部取长度 len 的最新子序列
     * @param out_types 输出：最新的 len 个节点类型
     * @param len 要求的序列长度（>= 2）
     * @return true 若 recent_sequence_ 中元素 >= len
     */
    bool getLatestSequenceTail(std::vector<std::string>& out_types, size_t len) const;

    /**
     * @brief 遍历所有可能的反向融合窗口（从长到短），尝试在 C3KernelRegistry 中命中一个
     * @param grad 下游梯度（取 shape）
     * @param first_forward_input_shape 首个 forward 输入形状
     * @param out_key 输出：命中的注册 key
     * @return true 若命中
     */
    bool tryLookupFusedBackwardKey(const Tensor& grad,
                                    const std::vector<size_t>& first_forward_input_shape,
                                    std::string& out_key);

    // ======================= 统计字段新增 =======================
    size_t fusion_hit_count_ = 0;
    size_t fusion_miss_count_ = 0;
    size_t mimo_compile_count_ = 0; ///< MIMO 编译次数
    size_t mimo_hit_count_ = 0;     ///< MIMO 执行命中次数
    size_t mimo_miss_count_ = 0;    ///< MIMO 尝试未命中次数
    uint64_t mimo_exec_ns_ = 0;     ///< MIMO kernel->execute() 累计耗时(ns, stats_mutex_ 保护)
    uint64_t mimo_keybuild_ns_ = 0; ///< MIMO cache key 构建累计耗时(ns, stats_mutex_ 保护)

    // ========== [P0.1 2026-08-30 苏璃珞] backward fallback 覆盖率统计 ==========
    // stats_mutex_ 保护；Stats::backward_fallback_reasons 是拷贝（map），保证 Stats 是 const-safe
    size_t backward_attempt_count_ = 0;             ///< tryExecuteBackward 总调用次数
    size_t backward_c3_attempt_count_ = 0;         ///< 走 C3 路径（compile + execute kernel）
    size_t backward_eager_fallback_count_ = 0;     ///< fallback 到 eager 的次数
    std::unordered_map<std::string, size_t> backward_fallback_reasons_;  ///< fallback 原因分类
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_BACKWARD_CAPTURE_H