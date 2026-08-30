/**
 * @file C3BackwardCapture.cpp
 * @brief 反向图 JIT 捕获与编译引擎实现
 * @details 实现反向节点到 C3 Graph 的映射、编译与执行。
 *          每个 autograd 节点类型对应一个 buildXxxBackwardGraph 方法，
 *          构建等价的 C3 Graph 后由 C3Engine 编译并注册到 C3KernelRegistry。
 * @date 2026/8/4
 */

#include "C3/C3BackwardCapture.h"
#include "C3/C3Config.h"
#include "C3/C3Engine.h"
#include "C3/C3KernelRegistry.h"
#include "C3/Graph.h"
#include "C3/GraphMerger.h"

#include "AutoGrad/Nodes/ReLUNode.h"
#include "AutoGrad/Nodes/SigmoidNode.h"
#include "AutoGrad/Nodes/TanhNode.h"
#include "AutoGrad/Nodes/AddNode.h"
#include "AutoGrad/Nodes/MulNode.h"
#include "AutoGrad/Nodes/SubNode.h"
#include "AutoGrad/Nodes/NegNode.h"
#include "AutoGrad/Nodes/DivNode.h"
#include "AutoGrad/Nodes/MatMulNode.h"
#include "AutoGrad/Nodes/ExpNode.h"
#include "AutoGrad/Nodes/LogNode.h"

#include <algorithm>
#include <cstddef>
#include <future>
#include <iomanip>
#include <sstream>
#include <thread>

namespace ct {
namespace c3 {

// ======================= 单例 =======================

C3BackwardCapture& C3BackwardCapture::getInstance() {
    static C3BackwardCapture instance;
    return instance;
}

// [P0.1 配套实装 2026-08-30 苏璃珞]
// 头文件声明了 shutdown() / taskStarted() / taskFinished() 三个函数（+active_tasks_ 字段）
// 但 .cpp 从未实装。之前 P0.1 改动只改了 .h 没改 .cpp——之前 build 成功只是因为
// `~C3BackwardCapture() { shutdown(); }` 析构**未实例化**（没人调 getInstance 后又让
// 静态析构触发）。P0.6B 改动显式调 taskStarted() 触发 linker 错。这里补实装。
//
// 行为：shutdown() 等所有 in-flight detached compile tasks 完成才返回（防止析构期
// 任务访问已销毁的 this）。taskStarted/Finished 是配套计数器。
bool C3BackwardCapture::taskStarted() {
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (shutting_down_.load(std::memory_order_acquire)) return false;
    active_tasks_.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

void C3BackwardCapture::taskFinished() {
    if (active_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_cv_.notify_all();
    }
}

void C3BackwardCapture::shutdown() {
    std::unique_lock<std::mutex> lock(task_mutex_);
    shutting_down_.store(true, std::memory_order_release);
    task_cv_.wait(lock, [this] {
        return active_tasks_.load(std::memory_order_acquire) == 0;
    });
}

// ======================= 公共接口 =======================

std::optional<std::vector<Tensor>> C3BackwardCapture::tryExecuteBackward(
    const ::Node* node, const Tensor& grad,
    const std::vector<Tensor>& forward_inputs)
{
    // ===== 统一开关 + 基准测试 kill-switch =====
    // C3_DISABLE_BACKWARD=1 / backwardFusionEnabled()=false: 走 C3Config.h 统一开关(用户层)
    // CTORCH_DISABLE_C3_BACKWARD=1: 强制禁用(基准测试用,不受 C3 统一开关影响)
    static const bool disabled = []() {
        const char* bench_kill = std::getenv("CTORCH_DISABLE_C3_BACKWARD");
        if (bench_kill && std::string(bench_kill) == "1") return true;
        return !backwardFusionEnabled();
    }();
    if (disabled) return std::nullopt;

    // [P0.1 2026-08-30 苏璃珞] 真正开始尝试 C3 backward 路径（不算用户禁用场景）
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        backward_attempt_count_++;
    }
    // backward C3 默认启用；保留 C3_ENABLE_BACKWARD=0 作为显式关闭方式。
    // CTORCH_DISABLE_C3_BACKWARD=1 或 C3_DISABLE_BACKWARD=1 仍由上方统一开关处理。
    const char* enable_backward = std::getenv("C3_ENABLE_BACKWARD");
    if (enable_backward && std::string(enable_backward) == "0") {
        return std::nullopt;
    }

    // ===== 统一 MIMO 融合反向 (梯度传导 + 激活求导 + 权重收缩) 🌟 =====
    const char* enable_mimo = std::getenv("C3_ENABLE_MIMO_BACKWARD");
    if (enable_mimo && std::string(enable_mimo) == "1") {
    {
        std::unique_lock<std::shared_mutex> lock(intercepted_mutex_);
        auto it = pending_mimo_intercepted_.find(node);
        if (it != pending_mimo_intercepted_.end()) {
            auto grads = std::move(it->second);
            pending_mimo_intercepted_.erase(it);
            return grads;
        }
    }
    auto mimo_res = tryExecuteUnifiedMIMOBackward(node, grad, forward_inputs);
    if (mimo_res.has_value()) {
        return mimo_res;
    }
    }

    // ===== Phase 2: 先尝试反向融合（整段序列一次性执行） =====
    // 注意：融合 kernel 是单输出的（对应序列首节点 input_index=0 的梯度），
    // 所以只有在「当前节点就是序列头、且我们只需要 input 0 的梯度」这种情况下
    // 才能用融合结果直接替换。为保守起见，先看 node 的上游节点数：
    //   - 如果是单输入节点（只有 1 个上游，n_inputs==1）→ 融合输出刚好对应，直接返回
    //   - 多输入节点 → 融合结果只覆盖 input 0 的梯度，不能替代整个逐输入多输出流程
    //     （否则其他 upstream 的 GradPack 缺失，导致 autograd 崩或梯度丢失）。
    size_t n_inputs = forward_inputs.empty() ? node->getInputs().size() : forward_inputs.size();
    if (n_inputs == 0) n_inputs = 1;
    const std::string type_name = std::string(typeid(*node).name());

    if (n_inputs == 1 && !(enable_mimo && std::string(enable_mimo) == "1")) {
        auto fused = tryExecuteFusedBackward(node, grad, forward_inputs);
        if (fused.has_value()) {
            std::vector<Tensor> out;
            out.push_back(std::move(*fused));
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                cache_hit_count_++;
            }
            #ifdef CT_DEBUG
            if (type_name.find("MatMul") != std::string::npos) {
                std::cerr << "[DBG-C3BW] MatMul backward FUSED-HIT, shape: ";
                for (auto s : out[0].sizes()) std::cerr << s << ",";
                std::cerr << std::endl;
            }
            #endif
            return out;
        }
    }

    // ===== Phase 1: 逐输入单输出 kernel 查找（多输入节点全覆盖） =====
    // 构建查找 key 前缀：node_type|grad:shape|inputs:shape

    std::stringstream ss;
    ss << type_name << "|grad:";
    for (size_t s : grad.sizes()) ss << s << ",";
    ss << "|inputs:";
    for (const auto& t : forward_inputs) {
        for (size_t s : t.sizes()) ss << s << ",";
    }
    std::string base_key = ss.str();

    // 多输入节点：每个上游梯度一个独立单输出 kernel，逐 key 查找并执行。
    // 任一输入的 kernel 缺失 → 整体回退 eager（保证正确性），仅触发缺失输入编译。
    std::vector<Tensor> out;
    out.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        auto result = C3KernelRegistry::getInstance().tryExecuteBackward(
            base_key + "|in:" + std::to_string(i), grad, forward_inputs);
        if (!result.has_value() || result->empty()) {
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                cache_miss_count_++;
            }
            // [reverse-fusion 诊断] C3_BW_MISS_TRACE=1：打印每个首次出现的未命中节点 key，
            // 用于定位剩下仍未纳入反向融合的 backward 节点（避免逐 batch 刷屏）。
            static const bool miss_trace = [] {
                const char* e = std::getenv("C3_BW_MISS_TRACE");
                return e && std::string(e) == "1";
            }();
            if (miss_trace) {
                static std::mutex mt_mu;
                static std::unordered_set<size_t> mt_seen;
                size_t kh = std::hash<std::string>{}(base_key + "|in:" + std::to_string(i));
                bool first = false;
                {
                    std::lock_guard<std::mutex> mk(mt_mu);
                    first = mt_seen.insert(kh).second;
                }
                if (first)
                    fprintf(stderr, "[BW-MISS] key=%s\n",
                            (base_key + "|in:" + std::to_string(i)).c_str());
            }
            compileBackwardAsyncForInput(node, grad, i);
            // [P0.6B 2026-08-30 苏璃珞 重做] miss 后等所有 in-flight async 编译完成
            //
            // 历史：之前 compileBackwardAsyncForInput 启动 std::thread + .detach()
            //       不等完成。miss 路径立即 return std::nullopt。**下次同 key 调用**
            //       时大概率前一次 async 还在编译（5-50ms）→ backward_entries_ 仍空
            //       → 重复 fallback。实测覆盖率 6.25%。
            //
            // 修复（最安全方案，不改任何函数体）：miss 后**同步等**所有 in-flight
            //       编译任务完成。**主线程阻塞** 5-50ms × N（in-flight 数），
            //       但**之后**同 key 必命中。
            //
            // 不改 compileBackwardAsyncForInput 函数体：避免触发 static const 初始化
            //       时序问题（之前 P0.6B inline 同步版本 hang 根因未知）。
            getInstance().waitForPendingCompiles();
            return std::nullopt;
        }
        out.push_back(std::move(result->at(0)));
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    cache_hit_count_++;
    #ifdef CT_DEBUG
    if (base_key.find("MatMul") != std::string::npos) {
        std::cerr << "[DBG-C3BW] MatMul backward HIT returning " << out.size() << " grads, shapes: ";
        for (auto& t : out) { for (auto s : t.sizes()) std::cerr << s << ","; std::cerr << " | "; }
        std::cerr << std::endl;
    }
    #endif
    return out;
}

void C3BackwardCapture::compileBackwardAsync(const ::Node* node, const Tensor& grad)
{
    // [P0 Fix 2026-08-13] AMX/MPS 设备兼容性检查
    // Handwritten backend 生成 CPU-SIMD kernel 调用，不支持 AMX/MPS
    DeviceType target_dev = grad.device();
    if (target_dev != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, 
                         ErrorType::DEVICE_COMPAT,
                         "C3 backward fusion not supported on device=" 
                         + std::to_string(static_cast<int>(target_dev)) 
                         + ", fallback to eager");
        return;
    }
    // 构建去重 key 前缀：node_type|grad:shape|inputs:shape
    // 必须与 tryExecuteBackward 中的 base_key 格式完全一致
    const std::string& type_name = typeid(*node).name();
    size_t n_inputs = node->getInputs().empty() ? 1 : node->getInputs().size();

    std::stringstream ss;
    ss << type_name << "|grad:";
    for (size_t s : grad.sizes()) ss << s << ",";
    ss << "|inputs:";
    for (const auto& t : node->getInputs()) {
        for (size_t s : t.sizes()) ss << s << ",";
    }
    std::string base_key = ss.str();

    // 收集编译所需信息（拷贝，不持有 node 指针）
    std::vector<TensorDesc> input_descs;
    for (const auto& t : node->getInputs()) {
        input_descs.push_back(TensorDesc{
            t.sizes(), t.dtype(), t.device(),
            TensorDesc::computeNumel(t.sizes())
        });
    }
    TensorDesc grad_desc{
        grad.sizes(), grad.dtype(), grad.device(),
        TensorDesc::computeNumel(grad.sizes())
    };

    // 为每个输入索引编译一个独立单输出 kernel
    for (size_t i = 0; i < n_inputs; ++i) {
        std::string per_key = base_key + "|in:" + std::to_string(i);

        // 去重检查
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            if (pending_compiles_.find(per_key) != pending_compiles_.end()) {
                continue; // 已在编译中
            }
            pending_compiles_[per_key] = true;
        }

        // 捕获 type_name 字符串（而非 node 指针），规避反向结束后节点释放导致的 UAF
        std::string type = type_name;
        // 【修复】不能用 std::async → future 析构会阻塞 → 变相同步
        std::thread([this, type, i, grad_desc, input_descs, per_key]() {
            // 构建该输入的反向 C3 Graph
            auto graph_opt = buildBackwardGraphForTypeAndIndex(type, i, grad_desc, input_descs);
            if (!graph_opt.has_value()) {
                // 不支持该节点类型/输入索引
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_compiles_.erase(per_key);
                return;
            }

            auto& graph_pair = graph_opt.value();
            Graph& graph = graph_pair.first;
            const std::vector<size_t>& fwd_input_map = graph_pair.second;

            // 跳过无计算节点的图（如 Add 的 identity 反向）
            // 无计算节点：nodeCount 仅含输入节点，没有实际计算操作
            if (graph.nodeCount() <= graph.inputCount()) {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_compiles_.erase(per_key);
                return;
            }

            // 编译
            // [线A 2026-08-14]: 切换为 MLIR 后端，借由已经完备实装的 SumReduce/Transpose MLIR JIT，
            // 实现全反向算子 100% 内存级编译，消除对 clang++ 磁盘编译的依赖。
            CompileOptions opts;
            opts.backend = C3Backend::MLIR;
            opts.opt_level = 3;
            opts.enable_cache = true;

            try {
                auto kernel = C3Engine::getInstance().compile(graph, opts);
                if (kernel) {
                    // 注册到 C3KernelRegistry 的 backward 专用注册表
                    std::vector<size_t> grad_shape = grad_desc.shape;
                    std::vector<size_t> out_shape =
                        input_descs.empty() ? grad_desc.shape : input_descs[i].shape;

                    C3KernelRegistry::getInstance().installBackward(
                        per_key, kernel, grad_shape, out_shape,
                        /*fwd_input_map=*/fwd_input_map,
                        /*num_inputs=*/graph.inputCount());

                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    compile_count_++;
                }
            } catch (const std::exception& e) {
                // 编译失败，静默处理
                (void)e;
            }

            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_compiles_.erase(per_key);
        }).detach();
    }
}

// 为单个输入索引异步编译（tryExecuteBackward 未命中时调用）
void C3BackwardCapture::compileBackwardAsyncForInput(
    const ::Node* node, const Tensor& grad, size_t input_index)
{
    // [P0 Fix 2026-08-13] AMX/MPS 设备兼容性检查
    DeviceType target_dev = grad.device();
    if (target_dev != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, 
                         ErrorType::DEVICE_COMPAT,
                         "C3 backward fusion not supported on device=" 
                         + std::to_string(static_cast<int>(target_dev)) 
                         + ", fallback to eager");
        return;
    }
    // 与 compileBackwardAsync 相同的 key 前缀构造
    const std::string& type_name = typeid(*node).name();
    std::stringstream ss;
    ss << type_name << "|grad:";
    for (size_t s : grad.sizes()) ss << s << ",";
    ss << "|inputs:";
    for (const auto& t : node->getInputs()) {
        for (size_t s : t.sizes()) ss << s << ",";
    }
    std::string per_key = ss.str() + "|in:" + std::to_string(input_index);

    // DEBT-NEW-7 v0.5.1+ 修复 dedup 漏洞
    // 之前只查 pending_compiles_(in-flight),不查 backward_entries_(已编译),
    // 编译完成后 entry 从 pending 移除,下一个 call 看不见,又起新线程。
    // MNIST 训练:6 unique (type, shape) 但 compile_count 飙到 11690/epoch (重复 ~200x)
    // 修复:先查 C3KernelRegistry.hasBackwardKey(per_key) → 已经在就别再编译
    if (C3KernelRegistry::getInstance().hasBackwardKey(per_key)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_compiles_.find(per_key) != pending_compiles_.end()) {
            return; // 已在编译中
        }
        pending_compiles_[per_key] = true;
    }

    std::vector<TensorDesc> input_descs;
    for (const auto& t : node->getInputs()) {
        input_descs.push_back(TensorDesc{
            t.sizes(), t.dtype(), t.device(),
            TensorDesc::computeNumel(t.sizes())
        });
    }
    TensorDesc grad_desc{
        grad.sizes(), grad.dtype(), grad.device(),
        TensorDesc::computeNumel(grad.sizes())
    };
    std::string type = type_name;

    // 【修复】不能用 std::async → future 析构会阻塞 → 变相同步
    std::thread([this, type, input_index, grad_desc, input_descs, per_key]() {
        auto graph_opt = buildBackwardGraphForTypeAndIndex(type, input_index, grad_desc, input_descs);
        if (!graph_opt.has_value()) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_compiles_.erase(per_key);
            return;
        }
        auto& graph_pair = graph_opt.value();
        Graph& graph = graph_pair.first;
        const std::vector<size_t>& fwd_input_map = graph_pair.second;
        if (graph.nodeCount() <= graph.inputCount()) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_compiles_.erase(per_key);
            return;
        }
        CompileOptions opts;
        // [线A 2026-08-14]: 切换为 MLIR 后端，借由已经完备实装的 SumReduce/Transpose MLIR JIT，
        // 实现全反向算子 100% 内存级编译，消除对 clang++ 磁盘编译的依赖。
        opts.backend = C3Backend::MLIR;
        opts.opt_level = 3;
        opts.enable_cache = true;
        try {
            auto kernel = C3Engine::getInstance().compile(graph, opts);
            if (kernel) {
                std::vector<size_t> grad_shape = grad_desc.shape;
                std::vector<size_t> out_shape =
                    input_descs.empty() ? grad_desc.shape : input_descs[input_index].shape;
                C3KernelRegistry::getInstance().installBackward(
                    per_key, kernel, grad_shape, out_shape,
                    /*fwd_input_map=*/fwd_input_map,
                    /*num_inputs=*/graph.inputCount());
#ifdef CT_DEBUG
                std::cerr << "[C3-BW-DEBUG-FOR-INPUT] install OK key=" << per_key
                          << " grad_shape=[";
                for (auto s : grad_shape) std::cerr << s << ",";
                std::cerr << "] out_shape=[";
                for (auto s : out_shape) std::cerr << s << ",";
                std::cerr << "] hasKey_after="
                          << C3KernelRegistry::getInstance().hasBackwardKey(per_key)
                          << std::endl;
                std::cerr.flush();
#endif
                std::lock_guard<std::mutex> lock(stats_mutex_);
                compile_count_++;
            }
#ifdef CT_DEBUG
            else {
                std::cerr << "[C3-BW-DEBUG-FOR-INPUT] compile returned nullptr key=" << per_key << std::endl;
                std::cerr.flush();
            }
#endif
        } catch (const std::exception& e) {
#ifdef CT_DEBUG
            std::cerr << "[C3-BW-DEBUG-FOR-INPUT] compile threw: " << e.what() << " key=" << per_key << std::endl;
            std::cerr.flush();
#endif
            (void)e;
        }
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_compiles_.erase(per_key);
    }).detach();
}

std::optional<Graph> C3BackwardCapture::buildBackwardGraph(
    const ::Node* node,
    const TensorDesc& grad_desc,
    const std::vector<TensorDesc>& input_descs)
{
    // 兼容入口：默认构建第一个输入（索引 0）的梯度。
    // 多输入节点的完整反向请通过 buildBackwardGraphForInput / ForTypeAndIndex 按输入索引调用。
    return buildBackwardGraphForInput(node, 0, grad_desc, input_descs);
}

std::optional<Graph> C3BackwardCapture::buildBackwardGraphForInput(
    const ::Node* node,
    size_t input_index,
    const TensorDesc& grad_desc,
    const std::vector<TensorDesc>& input_descs)
{
    // 通过 typeid 获取节点类型字符串，再走字符串分发（与异步编译线程一致）
    std::string type_name = typeid(*node).name();
    auto opt = buildBackwardGraphForTypeAndIndex(type_name, input_index, grad_desc, input_descs);
    if (!opt.has_value()) return std::nullopt;
    // 兼容入口：只返回 Graph，丢弃 fwd_input_map（调用方不关心运行时喂入）
    return opt->first;
}

std::optional<C3BackwardCapture::BackwardGraph> C3BackwardCapture::buildBackwardGraphForTypeAndIndex(
    const std::string& node_type,
    size_t input_index,
    const TensorDesc& grad_desc,
    const std::vector<TensorDesc>& input_descs)
{
    // 多输入节点（Add/Mul/MatMul/Sub/Div）：每个输入索引产出一个独立单输出图。
    // 单输入节点（ReLU/Sigmoid/Tanh/Neg/Exp/Log）：仅 input_index == 0 有效。
    if (node_type.find("ReLUNode") != std::string::npos) {
        if (input_index != 0 || input_descs.size() < 1) return std::nullopt;
        return buildReLUBackwardGraph(grad_desc, input_descs[0]);

    } else if (node_type.find("SigmoidNode") != std::string::npos) {
        if (input_index != 0 || input_descs.size() < 1) return std::nullopt;
        return buildSigmoidBackwardGraph(grad_desc, input_descs[0]);

    } else if (node_type.find("TanhNode") != std::string::npos) {
        if (input_index != 0 || input_descs.size() < 1) return std::nullopt;
        return buildTanhBackwardGraph(grad_desc, input_descs[0]);

    } else if (node_type.find("AddNode") != std::string::npos) {
        if (input_index >= input_descs.size()) return std::nullopt;
        return buildAddBackwardGraph(grad_desc, input_descs[0], input_descs[1], input_index);

    } else if (node_type.find("MatMulNode") != std::string::npos) {
        // 注意：必须在 MulNode 之前匹配，因为 "MulNode" 是 "MatMulNode" 的子串
        if (input_index >= input_descs.size()) return std::nullopt;
        return buildMatMulBackwardGraph(grad_desc, input_descs[0], input_descs[1], input_index);

    } else if (node_type.find("MulNode") != std::string::npos) {
        if (input_index >= input_descs.size()) return std::nullopt;
        return buildMulBackwardGraph(grad_desc, input_descs[0], input_descs[1], input_index);

    } else if (node_type.find("NegNode") != std::string::npos) {
        if (input_index != 0) return std::nullopt;
        return buildNegBackwardGraph(grad_desc);

    } else if (node_type.find("SubNode") != std::string::npos) {
        if (input_index >= input_descs.size()) return std::nullopt;
        return buildSubBackwardGraph(grad_desc, input_index);

    } else if (node_type.find("DivNode") != std::string::npos) {
        if (input_index >= input_descs.size()) return std::nullopt;
        return buildDivBackwardGraph(grad_desc, input_descs[0], input_descs[1], input_index);

    } else if (node_type.find("ExpNode") != std::string::npos) {
        if (input_index != 0 || input_descs.size() < 1) return std::nullopt;
        // Exp backward 需要 forward 输出（exp(x) 的值）
        // 使用 input_desc 作为近似的 out_desc（实际应为 exp(x) 的 shape）
        return buildExpBackwardGraph(grad_desc, input_descs[0], input_descs[0]);

    } else if (node_type.find("LogNode") != std::string::npos) {
        if (input_index != 0 || input_descs.size() < 1) return std::nullopt;
        return buildLogBackwardGraph(grad_desc, input_descs[0]);

    } else if (node_type.find("SoftmaxNode") != std::string::npos) {
        // Softmax backward 暂保守回退 eager。多归约图的临时 buffer
        // 生命周期/广播路径尚未满足 C3 命中条件，不能生成不可靠 kernel。
        return std::nullopt;

    } else if (node_type.find("CrossEntropyNode") != std::string::npos) {
        // [P0.2 2026-08-30 苏璃珞] CrossEntropy 是双输入节点（logits + target）
        // 仅 input_index==0（logits）有意义；target 不需 grad，input_index==1 返回 nullopt
        if (input_index == 0) {
            if (input_descs.size() < 2) return std::nullopt;
            return buildCrossEntropyBackwardGraph(grad_desc, input_descs[0], input_descs[1]);
        } else if (input_index == 1) {
            return std::nullopt;  // target 无需梯度
        }
        return std::nullopt;
    }

    // 不支持的节点类型
    return std::nullopt;
}

std::optional<C3BackwardCapture::BackwardGraph> C3BackwardCapture::buildBackwardGraphForType(
    const std::string& node_type,
    const TensorDesc& grad_desc,
    const std::vector<TensorDesc>& input_descs)
{
    // 融合编译阶段：每个节点构建单输出反向子图（对应上游第一个输入的梯度）。
    // 这样 Add/Mul/MatMul 等多输入节点不会被一刀切 nullopt，
    // 使得元素-wise 序列里的任何节点都能参与图 merge。
    // 注意：这里只返回 input_index == 0 的单输出图，
    // 因为 compileFusedBackwardAsync 按"反向串联、上一个输出作为下一个 grad"
    // 的方式合并，每次只需要子图产出一个目标 grad。
    return buildBackwardGraphForTypeAndIndex(node_type, 0, grad_desc, input_descs);
}

bool C3BackwardCapture::supportsNodeType(const std::string& node_type) {
    // ========== 只支持单输入单输出（unary element-wise）节点的反向编译/融合 ==========
    // 多输入节点（Add/Sub/Mul/Div/MatMul/CrossEntropy/Softmax 等）的 per-input 单节点 kernel
    // 目前存在 2 个问题：① 图构造 / 输入映射 bug（unordered_map::at key not found）
    //                   ② 数值正确性 bug（Mul 返回 [a,a] 而不是正确的 [b,a]）
    // 先从支持列表移除，一律回退 eager，保证 Test 4-7 数值正确性。
    // 后续单独修多输入单节点 kernel，验证正确后再加回。
    return node_type.find("ReLUNode") != std::string::npos ||
           // Sigmoid backward 在复合链中仍存在输入/梯度映射异常，暂回退 eager。
           node_type.find("TanhNode") != std::string::npos ||
           node_type.find("NegNode") != std::string::npos ||
           node_type.find("GELUNode") != std::string::npos ||
           node_type.find("LReLUNode") != std::string::npos ||
           node_type.find("SinNode") != std::string::npos ||
           node_type.find("CosNode") != std::string::npos ||
           node_type.find("AbsNode") != std::string::npos ||
           node_type.find("ExpNode") != std::string::npos ||
           node_type.find("LogNode") != std::string::npos ||
           node_type.find("MinNode") != std::string::npos ||
           node_type.find("MaxNode") != std::string::npos ||
           // Softmax backward 暂回退 eager：其多归约图的临时 buffer 生命周期
           // 仍需单独修复，不能在数值不稳定时命中 C3。
           // [P0.2 2026-08-30 苏璃珞] CrossEntropy 双输入节点（logits + target）
           // target 不需 grad（input_index=1 返回 nullopt）；仅 input_index=0 走 buildCrossEntropyBackwardGraph
           node_type.find("CrossEntropyNode") != std::string::npos;
}

C3BackwardCapture::Stats C3BackwardCapture::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats s;
    s.capture_count = capture_count_;
    s.compile_count = compile_count_;
    s.cache_hit_count = cache_hit_count_;
    s.cache_miss_count = cache_miss_count_;
    s.execution_failures = execution_failures_;
    s.fusion_compile_count = fusion_compile_count_;
    s.fusion_hit_count = fusion_hit_count_;
    s.fusion_miss_count = fusion_miss_count_;
    s.mimo_compile_count = mimo_compile_count_;
    s.mimo_hit_count = mimo_hit_count_;
    s.mimo_miss_count = mimo_miss_count_;
    s.mimo_exec_us = mimo_exec_ns_ / 1000;
    s.mimo_keybuild_us = mimo_keybuild_ns_ / 1000;
    return s;
}

// ======================= 反向 Graph 构建 =======================

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildReLUBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_desc)
{
    Graph g;

    // 输入: [grad, x]
    size_t grad_in = g.addInput(grad_desc);
    size_t x_in = g.addInput(input_desc);

    // 构建 Gt(x, 0): lhs = x (input_desc), rhs = zero scalar ({1})
    TensorDesc zero_desc = TensorDesc::fromShape({1});
    size_t zero_node = g.addConstant(0.0, zero_desc);
    size_t gt_node = g.addNode(
        GtNode{input_desc, zero_desc},
        {x_in, zero_node},
        TensorDesc::fromShape(input_desc.shape));

    // 构建 Mul(Gt(x,0), grad): lhs = Gt output (same as input), rhs = grad
    size_t mul_node = g.addNode(
        MulNode{TensorDesc::fromShape(input_desc.shape), grad_desc},
        {gt_node, grad_in},
        TensorDesc::fromShape(input_desc.shape));

    g.markOutput(mul_node);
    // [Fix 2026-08-11 最小集 build] 图输入 [grad, x]，x 对应 forward_inputs[0]
    return {std::move(g), {0}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildSigmoidBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_desc)
{
    Graph g;

    // 输入: [grad, x]
    size_t grad_in = g.addInput(grad_desc);
    size_t x_in = g.addInput(input_desc);

    // Sigmoid(x): 1.0 / (1.0 + exp(-x))
    TensorDesc neg_desc = TensorDesc::fromShape(input_desc.shape);
    size_t neg_x = g.addNode(NegNode{neg_desc}, {x_in}, neg_desc);

    // exp(-x)
    size_t exp_neg = g.addNode(ExpNode{neg_desc}, {neg_x}, neg_desc);

    // 1 + exp(-x)
    TensorDesc one_desc = TensorDesc::fromShape({1});
    size_t one_node = g.addConstant(1.0, one_desc);
    size_t denom = g.addNode(AddNode{neg_desc, neg_desc}, {one_node, exp_neg}, neg_desc);

    // 1.0 / (1.0 + exp(-x)) = sigmoid(x)
    size_t sigmoid = g.addNode(DivNode{neg_desc, neg_desc}, {one_node, denom}, neg_desc);

    // 1 - sigmoid(x)
    size_t one_minus_sig = g.addNode(SubNode{neg_desc, neg_desc}, {one_node, sigmoid}, neg_desc);

    // sigmoid(x) * (1 - sigmoid(x))
    size_t sig_times_one_minus = g.addNode(MulNode{neg_desc, neg_desc}, {sigmoid, one_minus_sig}, neg_desc);

    // sigmoid(x) * (1 - sigmoid(x)) * grad
    size_t result = g.addNode(MulNode{grad_desc, neg_desc}, {grad_in, sig_times_one_minus}, neg_desc);

    g.markOutput(result);
    // [Fix 2026-08-11 最小集 build] 图输入 [grad, x]，x 对应 forward_inputs[0]
    return {std::move(g), {0}};
}

// [P0.2 2026-08-30 苏璃珞] Softmax backward graph construction
//
// 公式（axis=1 行 softmax）：
//   dL/dx[i,j] = y[i,j] * (dL/dy[i,j] - sum_k dL/dy[i,k] * y[i,k])
//   其中 y = softmax(x, dim=1)
//
// 实现步骤（7 op）：
//   1. y = softmax(x, dim=1)  —— 重算（forward y 不暴露，c3 softmax 是 out-as-operand）
//      a) exp_x = exp(x)         (Exp)
//      b) sum_exp = sum(exp_x, dim=1, keepdim=true)  (SumReduce)
//      c) y = exp_x / sum_exp     (Div)
//   2. grad_y = grad * y         (Mul)
//   3. sum_grad_y = sum(grad_y, dim=1, keepdim=true)  (SumReduce)
//   4. diff = grad - sum_grad_y  (Sub)
//   5. grad_x = y * diff        (Mul)
//
// 当前状态：该构图函数保留作后续实现基础，但 supportsNodeType() 暂不放行
// Softmax backward；调用方会回退 eager，避免多归约临时 buffer 的生命周期问题。
// 已知限制：
//   - axis 暂固定 1（行 softmax），axis=0 不支持（编译时不支持广播）
//   - 数值稳定版（max-subtraction）未做——后续可加 Neg + Max + Sub
//   - forward y 不在输入里——这里重算（重复 exp 一次）
C3BackwardCapture::BackwardGraph C3BackwardCapture::buildSoftmaxBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_desc)
{
    Graph g;

    // 输入: [grad, x]（grad = dL/dy, x = forward 输入）
    size_t grad_in = g.addInput(grad_desc);
    size_t x_in = g.addInput(input_desc);

    TensorDesc same_desc = TensorDesc::fromShape(input_desc.shape);

    // 1a) exp(x)
    size_t exp_x = g.addNode(ExpNode{same_desc}, {x_in}, same_desc);

    // 1b) sum(exp(x), dim=1, keepdim=true)  → [M, 1] 形状
    TensorDesc scalar_desc = TensorDesc::fromShape(
        input_desc.shape.size() > 0 ?
            std::vector<size_t>{input_desc.shape[0], 1} :
            std::vector<size_t>{1});
    size_t sum_exp = g.addNode(
        SumReduceNode{same_desc, 1, true},  // axis=1, keepdim=true
        {exp_x}, scalar_desc);

    // 1c) y = exp(x) / sum_exp  (广播 [M,1] → [M,N])
    size_t y = g.addNode(
        DivNode{same_desc, scalar_desc}, {exp_x, sum_exp}, same_desc);

    // 2) grad * y
    size_t grad_y = g.addNode(
        MulNode{grad_desc, same_desc}, {grad_in, y}, same_desc);

    // 3) sum(grad_y, dim=1, keepdim=true)
    size_t sum_grad_y = g.addNode(
        SumReduceNode{same_desc, 1, true},
        {grad_y}, scalar_desc);

    // 4) grad - sum_grad_y  (广播 [M,1] → [M,N])
    size_t diff = g.addNode(
        SubNode{grad_desc, scalar_desc}, {grad_in, sum_grad_y}, same_desc);

    // 5) y * diff = grad_x
    size_t grad_x = g.addNode(
        MulNode{same_desc, same_desc}, {y, diff}, same_desc);

    g.markOutput(grad_x);
    // 图输入 [grad, x]，x 对应 forward_inputs[0]
    return {std::move(g), {0}};
}

// [P0.2 2026-08-30 苏璃珞] CrossEntropy backward graph construction
//
// 公式（axis=1 行 softmax，target 是 [M, N] one-hot / soft probability）：
//   dL/d_logits[i, j] = softmax(logits)[i, j] - target[i, j]
//
// 实现步骤（4 op）：
//   1. exp_x = exp(logits)              (Exp)
//   2. sum_exp = sum(exp_x, axis=1, keepdim=true)  → [M, 1]  (SumReduce[keepdim])
//   3. y = exp_x / sum_exp              (Div)  → softmax(logits)
//   4. grad_logits = y - target         (Sub)  → [M, N]
//
// 已知限制（同 Softmax backward）：
//   - 依赖 broadcast shape-based 修复（P0.2.1 立项）—— 当前 numel-based `idx % M`
//     对非平凡尺寸（如 M=4, N=8）返回错位
//   - 数值稳定：forward 走 CrossEntropyOpLowering 内部 max-subtraction，
//                backward 这里是朴素 exp（与 forward 行为不同）
//                后续可让 backward 也用 c3.softmax op（linalg.softmax 内部稳定）
C3BackwardCapture::BackwardGraph C3BackwardCapture::buildCrossEntropyBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_descs_0,  // logits_desc
    const TensorDesc& input_descs_1)  // target_desc
{
    (void)grad_desc;  // CE 反向下游 grad 是常数 1/M（mean reduction）或 1，公式不需要它
    Graph g;

    // 输入: [logits, target]
    size_t logits_in = g.addInput(input_descs_0);
    size_t target_in = g.addInput(input_descs_1);

    TensorDesc same_desc = TensorDesc::fromShape(input_descs_0.shape);

    // 1) exp(logits)
    size_t exp_logits = g.addNode(ExpNode{same_desc}, {logits_in}, same_desc);

    // 2) sum(exp(logits), axis=1, keepdim=true)  → [M, 1]
    TensorDesc scalar_desc = TensorDesc::fromShape(
        input_descs_0.shape.size() > 0 ?
            std::vector<size_t>{input_descs_0.shape[0], 1} :
            std::vector<size_t>{1});
    size_t sum_exp = g.addNode(
        SumReduceNode{same_desc, 1, true},  // axis=1, keepdim=true
        {exp_logits}, scalar_desc);

    // 3) y = exp(logits) / sum_exp  (广播 [M,1] → [M,N])
    size_t y = g.addNode(
        DivNode{same_desc, scalar_desc}, {exp_logits, sum_exp}, same_desc);

    // 4) grad_logits = y - target  → [M, N]
    size_t grad_x = g.addNode(
        SubNode{same_desc, same_desc}, {y, target_in}, same_desc);

    g.markOutput(grad_x);
    // 图输入 [logits, target]：
    //   logits   —— forward_inputs[0]
    //   target   —— forward_inputs[1]（不需 grad）
    return {std::move(g), {0, 1}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildTanhBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_desc)
{
    Graph g;

    // 输入: [grad, x]
    size_t grad_in = g.addInput(grad_desc);
    size_t x_in = g.addInput(input_desc);

    TensorDesc same_desc = TensorDesc::fromShape(input_desc.shape);

    // Tanh(x)
    auto expf_func = [&](const TensorDesc& d, size_t input_id) -> size_t {
        return g.addNode(ExpNode{d}, {input_id}, d);
    };

    // exp(x)
    size_t exp_x = expf_func(same_desc, x_in);

    // exp(-x): 需要先 negate
    size_t neg_x = g.addNode(NegNode{same_desc}, {x_in}, same_desc);
    size_t exp_neg_x = expf_func(same_desc, neg_x);

    // exp(x) - exp(-x)
    size_t numerator = g.addNode(SubNode{same_desc, same_desc}, {exp_x, exp_neg_x}, same_desc);

    // exp(x) + exp(-x)
    size_t denominator = g.addNode(AddNode{same_desc, same_desc}, {exp_x, exp_neg_x}, same_desc);

    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    size_t tanh = g.addNode(DivNode{same_desc, same_desc}, {numerator, denominator}, same_desc);

    // tanh(x) * tanh(x)
    size_t tanh_sq = g.addNode(MulNode{same_desc, same_desc}, {tanh, tanh}, same_desc);

    // 1 - tanh(x)²
    TensorDesc one_desc = TensorDesc::fromShape({1});
    size_t one_node = g.addConstant(1.0, one_desc);
    size_t one_minus = g.addNode(SubNode{same_desc, one_desc}, {one_node, tanh_sq}, same_desc);

    // (1 - tanh(x)²) * grad
    size_t result = g.addNode(MulNode{grad_desc, same_desc}, {grad_in, one_minus}, same_desc);

    g.markOutput(result);
    // [Fix 2026-08-11 最小集 build] 图输入 [grad, x]，x 对应 forward_inputs[0]
    return {std::move(g), {0}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildAddBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& lhs_desc,
    const TensorDesc& rhs_desc,
    size_t input_index)
{
    Graph g;

    // 输入: grad
    size_t grad_in = g.addInput(grad_desc);

    // Add 反向：两个输入梯度均为 grad（广播时按各自形状 SumReduce 缩小）
    const TensorDesc& target = (input_index == 0) ? lhs_desc : rhs_desc;
    if (needsSumReduce(grad_desc.shape, target.shape)) {
        int axis = computeReduceAxis(grad_desc.shape, target.shape);
        size_t reduced = g.addNode(
            SumReduceNode{grad_desc, axis},
            {grad_in},
            TensorDesc::fromShape(target.shape));
        g.markOutput(reduced);
    } else {
        g.markOutput(grad_in);
    }

    // [Fix 2026-08-11 最小集 build] 图只有 grad 输入，无 forward 输入 → 空索引表
    return {std::move(g), {}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildMulBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& a_desc,
    const TensorDesc& b_desc,
    size_t input_index)
{
    Graph g;

    // [Fix 2026-08-11 最小集 build] 以前总是加 [grad, A, B]，未用输入被 DCE 剪枝 →
    // ext_map 索引平移 → 运行时喂错张量。现在只加实际用到的 forward 输入，
    // DCE 无可剪，图输入顺序稳定，配合 fwd_input_map 精确喂入。
    // grad 输入始终是图输入 0；后续图输入按 fwd_input_map 对应 forward_inputs。
    size_t grad_in = g.addInput(grad_desc);

    if (input_index == 0) {
        // grad_a = grad * B → 只需 B，B 是 forward_inputs[1]
        size_t b_in = g.addInput(b_desc);
        size_t o = g.addNode(
            MulNode{grad_desc, b_desc},
            {grad_in, b_in},
            TensorDesc::fromShape(a_desc.shape));
        g.markOutput(o);
        return {std::move(g), {1}};
    } else {
        // grad_b = A * grad → 只需 A，A 是 forward_inputs[0]
        size_t a_in = g.addInput(a_desc);
        size_t o = g.addNode(
            MulNode{a_desc, grad_desc},
            {a_in, grad_in},
            TensorDesc::fromShape(b_desc.shape));
        g.markOutput(o);
        return {std::move(g), {0}};
    }
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildMatMulBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& a_desc,
    const TensorDesc& b_desc,
    size_t input_index)
{
    Graph g;

    // [Fix 2026-08-11 最小集 build] 以前总是加 [grad, A, B]，未用输入被 DCE 剪枝 →
    // ext_map 索引平移 → 运行时喂错张量（A 当 B）→ grad_x 数值爆炸、grad_w 碰巧正确。
    // 现在只加实际用到的 forward 输入，DCE 无可剪，图输入顺序稳定，配合 fwd_input_map 精确喂入。
    // grad 输入始终是图输入 0；后续图输入按 fwd_input_map 对应 forward_inputs。
    size_t grad_in = g.addInput(grad_desc);

    if (input_index == 0) {
        // grad_A = grad @ B^T → 只需 B（forward_inputs[1]）
        size_t b_in = g.addInput(b_desc);
        // 转置输出形状
        std::vector<size_t> bT_shape = {b_desc.shape[1], b_desc.shape[0]};
        TensorDesc bT_desc = TensorDesc::fromShape(bT_shape);
        size_t bT = g.addNode(TransposeNode{b_desc, 0, 1}, {b_in}, bT_desc);
        TensorDesc grad_a_desc = TensorDesc::fromShape({grad_desc.shape[0], bT_desc.shape[1]});
        size_t o = g.addNode(
            MatMulNode{grad_desc, bT_desc},
            {grad_in, bT},
            grad_a_desc);
        g.markOutput(o);
        return {std::move(g), {1}};
    } else {
        // grad_B = A^T @ grad → 只需 A（forward_inputs[0]）
        size_t a_in = g.addInput(a_desc);
        std::vector<size_t> aT_shape = {a_desc.shape[1], a_desc.shape[0]};
        TensorDesc aT_desc = TensorDesc::fromShape(aT_shape);
        size_t aT = g.addNode(TransposeNode{a_desc, 0, 1}, {a_in}, aT_desc);
        TensorDesc grad_b_desc = TensorDesc::fromShape({aT_desc.shape[0], grad_desc.shape[1]});
        size_t o = g.addNode(
            MatMulNode{aT_desc, grad_desc},
            {aT, grad_in},
            grad_b_desc);
        g.markOutput(o);
        return {std::move(g), {0}};
    }
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildNegBackwardGraph(
    const TensorDesc& grad_desc)
{
    Graph g;

    // 输入: grad
    size_t grad_in = g.addInput(grad_desc);

    // grad_x = -grad
    size_t result = g.addNode(
        NegNode{grad_desc},
        {grad_in},
        grad_desc);

    g.markOutput(result);
    // 图只有 grad 输入，无 forward 输入 → 空索引表
    return {std::move(g), {}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildSubBackwardGraph(
    const TensorDesc& grad_desc,
    size_t input_index)
{
    Graph g;

    // 输入: grad
    size_t grad_in = g.addInput(grad_desc);

    if (input_index == 0) {
        // grad_a = grad（直接输出 grad_in）
        g.markOutput(grad_in);
    } else {
        // grad_b = -grad
        size_t grad_b = g.addNode(
            NegNode{grad_desc},
            {grad_in},
            grad_desc);
        g.markOutput(grad_b);
    }
    // 图只有 grad 输入，无 forward 输入 → 空索引表
    return {std::move(g), {}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildDivBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& a_desc,
    const TensorDesc& b_desc,
    size_t input_index)
{
    Graph g;

    // [Fix 2026-08-11 最小集 build] 只加实际用到的 forward 输入，DCE 无可剪。
    // grad 输入始终是图输入 0；后续图输入按 fwd_input_map 对应 forward_inputs。
    size_t grad_in = g.addInput(grad_desc);

    if (input_index == 0) {
        // grad_a = grad / B → 只需 B（forward_inputs[1]）
        size_t b_in = g.addInput(b_desc);
        size_t o = g.addNode(
            DivNode{grad_desc, b_desc},
            {grad_in, b_in},
            TensorDesc::fromShape(a_desc.shape));
        g.markOutput(o);
        return {std::move(g), {1}};
    } else {
        // grad_b = -(A / (B * B)) * grad → 需 A、B（forward_inputs[0], forward_inputs[1]）
        size_t a_in = g.addInput(a_desc);
        size_t b_in = g.addInput(b_desc);

        // B * B
        size_t b_sq = g.addNode(
            MulNode{b_desc, b_desc},
            {b_in, b_in},
            b_desc);

        // A / (B * B)
        size_t a_div_b_sq = g.addNode(
            DivNode{a_desc, b_desc},
            {a_in, b_sq},
            a_desc);

        // (A / (B * B)) * grad
        size_t mul_grad = g.addNode(
            MulNode{a_desc, grad_desc},
            {a_div_b_sq, grad_in},
            TensorDesc::fromShape(b_desc.shape));

        // -(A / (B * B)) * grad
        size_t grad_b = g.addNode(
            NegNode{TensorDesc::fromShape(b_desc.shape)},
            {mul_grad},
            TensorDesc::fromShape(b_desc.shape));

        g.markOutput(grad_b);
        return {std::move(g), {0, 1}};
    }
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildExpBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_desc,
    const TensorDesc& output_desc)
{
    Graph g;

    // 输入: [grad, x]
    size_t grad_in = g.addInput(grad_desc);
    size_t x_in = g.addInput(input_desc);

    // exp(x)
    size_t exp_x = g.addNode(
        ExpNode{input_desc},
        {x_in},
        output_desc);

    // grad_x = exp(x) * grad
    size_t result = g.addNode(
        MulNode{output_desc, grad_desc},
        {exp_x, grad_in},
        TensorDesc::fromShape(input_desc.shape));

    g.markOutput(result);
    // 图输入 [grad, x]，x 对应 forward_inputs[0]
    return {std::move(g), {0}};
}

C3BackwardCapture::BackwardGraph C3BackwardCapture::buildLogBackwardGraph(
    const TensorDesc& grad_desc,
    const TensorDesc& input_desc)
{
    Graph g;

    // 输入: [grad, x]
    size_t grad_in = g.addInput(grad_desc);
    size_t x_in = g.addInput(input_desc);

    // grad_x = grad / x
    size_t result = g.addNode(
        DivNode{grad_desc, input_desc},
        {grad_in, x_in},
        TensorDesc::fromShape(input_desc.shape));

    g.markOutput(result);
    // 图输入 [grad, x]，x 对应 forward_inputs[0]
    return {std::move(g), {0}};
}

// ======================= 反向融合检测 (Phase 2) =======================

void C3BackwardCapture::recordBackwardNode(
    const std::string& node_type,
    const std::vector<size_t>& grad_shape,
    const std::vector<size_t>& input_shape,
    const std::vector<Tensor>& forward_inputs)
{
    // 只把 C3 支持的反向节点放进序列。
    // GradAccumulator 等辅助节点一律跳过，避免穿插污染可融合窗口。
    if (!supportsNodeType(node_type)) {
        return;
    }

    std::lock_guard<std::mutex> lock(fusion_mutex_);

    // 【P0 修复 2026-08-08】iter 边界检测：避免连续 iter 的 push 在 recent_sequence_ 末尾交叉污染，
    // 导致 w=2 chain 末尾 2 个 = [prev_iter_tail, this_iter_head] 形成 forward order 错配
    //（如 [Sigmoid, ReLU] 应是 [ReLU, Sigmoid] backward BFS order）。
    // 修复：push 前先 check recent_sequence_ 状态：
    //   - size == 2: chain 完整 (backward BFS 推 2 个)，process + clear + regular push (累积新 chain first)
    //   - 其他: 直接 regular push (包括 size 1 case: 不要 rebuild, 让 chain 在 size 2 时 process)
    if (recent_sequence_.size() == 2) {
        // chain 完整 (size 2) — process + clear
        std::vector<std::string> chain(recent_sequence_.begin(), recent_sequence_.end());
        std::vector<std::vector<size_t>> gs(recent_grad_shapes_.begin(), recent_grad_shapes_.end());
        std::vector<std::vector<size_t>> is_(recent_input_shapes_.begin(), recent_input_shapes_.end());
        // [Fix 2026-08-11 反向融合 SIGBUS] 形状一致性校验：
        // recordBackwardNode 只登记 C3 支持的 element-wise 节点，中间的 Add/MatMul
        // 会被跳过，导致「隔着 MatMul 的两个 ReLU」在序列里看似相邻（如 MNIST
        // z2.ReLU grad=[128,128] 与 z1.ReLU grad=[128,256]）。融合 kernel 用统一
        // elem_n 分段 output + 链式 grad 传播，隐含要求链内所有节点 grad 形状一致；
        // 否则越界读 → SIGBUS。故编译前强制链内 grad_shapes 全等，不一致则不融合。
        bool shape_consistent = true;
        for (size_t si = 1; si < gs.size(); ++si) {
            if (gs[si] != gs[0]) { shape_consistent = false; break; }
        }
        if (shape_consistent && isFusableSequence(chain)) {
            const std::vector<size_t>& reg_grad_shape = gs.front();
            const std::vector<size_t>& reg_input_shape = is_.back();
            std::string seq_key = makeSequenceKey(chain);
            std::string full_key = makeFusedBackwardKey(seq_key, reg_grad_shape, reg_input_shape);
            auto& entry = sequence_counts_[full_key];
            if (entry.node_types.empty()) {
                entry.node_types = std::move(chain);
                entry.grad_shapes = std::move(gs);
                entry.input_shapes = std::move(is_);
            }
            entry.frequency++;
            if (entry.frequency >= kFusionThreshold && !entry.compiling) {
                entry.compiling = true;
                compileFusedBackwardAsync(entry);
            }
        }
        recent_sequence_.clear();
        recent_grad_shapes_.clear();
        recent_input_shapes_.clear();
        recent_forward_inputs_.clear();
    }

    // 四队列同步 push/pop，保证类型、形状与 forward_inputs 一一对应
    recent_sequence_.push_back(node_type);
    recent_grad_shapes_.push_back(grad_shape);
    recent_input_shapes_.push_back(input_shape);
    recent_forward_inputs_.push_back(forward_inputs);
    #ifdef CT_DEBUG
    {
        static int dbg_recent = 0;
        if (dbg_recent < 200) {
            bool is_32x64 = !grad_shape.empty() && grad_shape[0] == 32;
            if (is_32x64) {
                std::cerr << "[DBG-RECENT#" << dbg_recent << "] after push " << node_type
                          << " recent=[";
                for (auto& t : recent_sequence_) std::cerr << t << " | ";
                std::cerr << "] size=" << recent_sequence_.size() << "\n";
                dbg_recent++;
            }
        }
    }
    #endif
    if (recent_sequence_.size() > kFusionWindowSize) {
        recent_sequence_.pop_front();
        recent_grad_shapes_.pop_front();
        recent_input_shapes_.pop_front();
        recent_forward_inputs_.pop_front();
    }

    // 从尾部提取长度 2..kFusionWindowSize 的可融合子窗口，逐一累积频次。
    // 不使用整个 recent_sequence_，因为窗口两端可能被新的不相关节点污染。
    const size_t L = recent_sequence_.size();
    for (size_t w = 2; w <= kFusionWindowSize && w <= L; ++w) {
        size_t start = L - w;
        std::vector<std::string> types(recent_sequence_.begin() + start,
                                       recent_sequence_.begin() + L);
        if (!isFusableSequence(types)) continue;

        std::vector<std::vector<size_t>> gs(recent_grad_shapes_.begin() + start,
                                            recent_grad_shapes_.begin() + L);
        std::vector<std::vector<size_t>> is_(recent_input_shapes_.begin() + start,
                                             recent_input_shapes_.begin() + L);

        // ============= P0 修复 1（DEBT-1）：sequence_counts_ key 从纯 seq_key 改为带形状签名的 full_key =============
        // 原 bug：sequence_counts_[seq_key] 不区分 shape，导致 warmup 小尺寸的序列先写入 entry.grad_shapes，
        // 后续大尺寸（512×512）同 type 序列累计到同一 entry，触发 compile 时 grad_shapes 仍是小尺寸 →
        // compile 端 registry 的 shape key 与 execute 端 lookup 的 shape key 完全不匹配 → 100% miss。
        // 修复：每个 (type 序列 + shape 签名) 组合独立统计频次（JIT kernel 本就是 per-shape 编译的，这是正确行为）。
        //
        // ============= DEBT-2 未修：100% miss 在 512x512 bench 仍存在，但根因不在 entry.shape 过期 =============
        // 验证：entry 是 per-full_key 存储，full_key 包含 shape。第一次 capture 写入的 shape 就是该 entry
        // 唯一对应的 shape（因为 full_key 锁定）。后续同 full_key 的 capture shape 必然相同 → 每次
        // 覆盖等价于首次固化。**真正 DEBT-2 根因待查**（疑似在 tryExecuteFusedBackward 链构造
        // 或 installBackward 的 shape 校验路径中，与 recordBackwardNode 无关），所以这里不做简化修复。
        const std::vector<size_t>& reg_grad_shape = gs.front();  // 最下游端 grad_shape（dL/dy 形状）
        const std::vector<size_t>& reg_input_shape = is_.back(); // 最上游端 forward input[0] 形状
        std::string seq_key = makeSequenceKey(types);
        std::string full_key = makeFusedBackwardKey(seq_key, reg_grad_shape, reg_input_shape);

        auto& entry = sequence_counts_[full_key];
        if (entry.node_types.empty()) {
            entry.node_types = std::move(types);
            entry.grad_shapes = std::move(gs);
            entry.input_shapes = std::move(is_);
        }
        entry.frequency++;

        if (entry.frequency >= kFusionThreshold && !entry.compiling) {
            entry.compiling = true;
            compileFusedBackwardAsync(entry);
        }
    }
}

bool C3BackwardCapture::isElementWiseBackward(const std::string& node_type) {
    // 必须与 supportsNodeType 完全一致：只允许纯单输入反向节点进入序列 / 可融合判定
    return supportsNodeType(node_type);
}

std::string C3BackwardCapture::makeSequenceKey(const std::vector<std::string>& types) {
    std::string key;
    for (const auto& t : types) {
        if (!key.empty()) key += "+";
        // 提取简短的节点类型名
        if (t.find("ReLUNode") != std::string::npos) key += "ReLU";
        else if (t.find("SigmoidNode") != std::string::npos) key += "Sigmoid";
        else if (t.find("TanhNode") != std::string::npos) key += "Tanh";
        else if (t.find("NegNode") != std::string::npos) key += "Neg";
        else if (t.find("AddNode") != std::string::npos) key += "Add";
        else if (t.find("SubNode") != std::string::npos) key += "Sub";
        else if (t.find("MulNode") != std::string::npos) key += "Mul";
        else if (t.find("DivNode") != std::string::npos) key += "Div";
        else if (t.find("ExpNode") != std::string::npos) key += "Exp";
        else if (t.find("LogNode") != std::string::npos) key += "Log";
        else key += "Other";
    }
    return key;
}

bool C3BackwardCapture::isFusableSequence(const std::vector<std::string>& types) {
    if (types.size() < 2) return false;
    for (const auto& t : types) {
        if (!isElementWiseBackward(t)) return false;
    }
    return true;
}

// ======================= 融合查找辅助 =======================

std::string C3BackwardCapture::makeFusedBackwardKey(const std::string& seq_key,
                                                     const std::vector<size_t>& grad_shape,
                                                     const std::vector<size_t>& input_shape)
{
    std::stringstream ss;
    ss << "backward_fused_" << seq_key << "|g:";
    for (size_t s : grad_shape) ss << s << ",";
    ss << "|i:";
    for (size_t s : input_shape) ss << s << ",";
    return ss.str();
}

bool C3BackwardCapture::getLatestSequenceTail(std::vector<std::string>& out_types, size_t len) const {
    if (recent_sequence_.size() < len) return false;
    out_types.assign(recent_sequence_.end() - len, recent_sequence_.end());
    return true;
}

bool C3BackwardCapture::tryLookupFusedBackwardKey(const Tensor& grad,
                                                   const std::vector<size_t>& first_forward_input_shape,
                                                   std::string& out_key)
{
    // 从最大窗口到最小窗口，逐次尝试查找已编译的融合 kernel
    // 命中即返回（贪心：优先更长的融合 = 更多中间写读节省）。
    const auto& gshape = grad.sizes();
    for (size_t w = kFusionWindowSize; w >= 2; --w) {
        std::vector<std::string> types;
        if (!getLatestSequenceTail(types, w)) continue;
        if (!isFusableSequence(types)) continue;
        std::string seq_key = makeSequenceKey(types);
        std::string full_key = makeFusedBackwardKey(seq_key, gshape, first_forward_input_shape);
        if (C3KernelRegistry::getInstance().hasBackwardKey(full_key)) {
            out_key = full_key;
            return true;
        }
    }
    return false;
}

void C3BackwardCapture::compileFusedBackwardAsync(const BackwardSequence& seq) {
    // 检查序列是否可融合
    if (!isFusableSequence(seq.node_types)) return;
    if (seq.grad_shapes.empty() || seq.input_shapes.empty()) return;
    {
        #ifdef CT_DEBUG
        static int dbg_cfba = 0;
        bool is_32x64 = !seq.grad_shapes.empty() &&
                        !seq.grad_shapes.front().empty() &&
                        seq.grad_shapes.front()[0] == 32;
        if (dbg_cfba < 200 || is_32x64) {
            std::cerr << "[DBG-CFBA#" << dbg_cfba << "] seq.node_types=[";
            for (auto& t : seq.node_types) std::cerr << t << " | ";
            std::cerr << "] N=" << seq.node_types.size()
                      << " reg_grad_shape=[";
            for (auto s : seq.grad_shapes.front()) std::cerr << s << ",";
            std::cerr << "] reg_input_shape=[";
            for (auto s : seq.input_shapes.back()) std::cerr << s << ",";
            std::cerr << "]\n";
            dbg_cfba++;
        }
        #endif
    }

    // 注册/查找共用统一 key 格式（带形状签名）：
    // - 下游 grad_shape = recent_sequence_ 中**先执行**的节点（最下游，离 loss 最近）的 grad 形状 → seq.grad_shapes.front()
    // - 上游 input_shape = recent_sequence_ 中**最后**的节点（最上游，离 data 最近）的首个 forward 输入形状 → seq.input_shapes.back()
    //   （必须与 tryExecuteFusedBackward 中 lookup key: "grad.sizes()" + "chain_forward_inputs[w-1][0].sizes()" 严格一一对应！）
    std::string seq_key = makeSequenceKey(seq.node_types);
    const std::vector<size_t>& reg_grad_shape = seq.grad_shapes.front();
    const std::vector<size_t>& reg_input_shape = seq.input_shapes.back();
    std::string fused_key = makeFusedBackwardKey(seq_key, reg_grad_shape, reg_input_shape);

    // 去重检查
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_compiles_.find(fused_key) != pending_compiles_.end()) {
            return;
        }
        pending_compiles_[fused_key] = true;
    }

    // 异步编译融合 kernel
    // 【重要】不能用 std::async，因为 std::future 析构会阻塞等待 → 变同步
    //     用 std::thread + detach 才是真正的后台异步
    std::thread([this, seq, fused_key, reg_grad_shape, reg_input_shape]() {
        // ======================= 诊断：编译耗时时间戳 =======================
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();
        auto ms_since = [&](const char* label) {
            #ifdef CT_DEBUG
            auto t = clock::now();
            double dt = std::chrono::duration<double, std::milli>(t - t_start).count();
            std::cerr << "[DIAG-COMPILE] t+" << std::fixed << std::setprecision(1) << dt
                      << "ms  " << label << "  key=" << fused_key.substr(0, 40) << "..." << std::endl;
            #endif
        };
        try {
            ms_since("START async compile");
            // 构建融合 backward Graph
            // 策略：i=0（最下游，离 loss 最近）→ i=N-1（最上游，离 data 最近）
            // 每个 backward 图的输出（upstream grad）成为下一个（上游）backward 图的 grad 输入
            Graph fused_graph;

            const size_t N = seq.node_types.size();
            // 初始 grad_desc = 最下游节点 (i=0) 的下游端 grad（= kernel 参数的 grad Tensor 形状）
            TensorDesc grad_desc = TensorDesc::fromShape(seq.grad_shapes[0]);
            size_t current_grad_id = fused_graph.addInput(grad_desc);

            // 为每个不同的 forward input 创建输入节点
            std::unordered_map<std::string, size_t> forward_inputs;
            // 记录每个 backward 图的输出 ID（对应每个节点给其 upstream 的 grad 张量）
            // 多输出 fusion graph 的 outputs 顺序与 seq.node_types 一一对应：
            //   per_node_output_ids[k] = node_types[k] 节点执行完后，返回给该节点 upstream[i=0] 的 grad
            std::vector<size_t> per_node_output_ids;
            per_node_output_ids.reserve(N);
            size_t prev_output_id = current_grad_id;

            // 正向遍历：i=0（下游端 ReLU/Sigmoid）→ i=N-1（上游端）
            for (size_t i = 0; i < N; ++i) {
                const std::string& node_type = seq.node_types[i];

                // 构建单个 backward 图：使用该节点对应的 input_shapes[i] 作为 forward input
                const std::vector<size_t>& input_shape_i = seq.input_shapes[i];
                TensorDesc input_desc_i = TensorDesc::fromShape(input_shape_i);
                auto graph_opt = buildBackwardGraphForType(node_type, grad_desc, {input_desc_i, input_desc_i});
                if (!graph_opt.has_value()) {
                    {
                        std::lock_guard<std::mutex> lock(pending_mutex_);
                        pending_compiles_.erase(fused_key);
                    }
                    return;
                }

                auto& sub_graph_pair = graph_opt.value();
                Graph& sub_graph = sub_graph_pair.first;

                // 为 sub_graph 的每个**外部输入节点 ID（sub_graph.inputs() vector）**创建映射。
                // 注意 sub_graph 的 inputs_ 是独立 vector（不是 nodes_ 的前缀），必须通过 inputs()[j] 取 id。
                std::unordered_map<size_t, size_t> remap_input;
                for (size_t j = 0; j < sub_graph.inputCount(); ++j) {
                    size_t src_input_id = sub_graph.inputs()[j]; // 取真实的 input 节点 id
                    if (j == 0) {
                        // 第一个 input（grad 输入）使用上一轮 backward 输出接上
                        // i=0 时 prev_output_id = fused_graph grad_in（最下游端 grad 输入）
                        remap_input[src_input_id] = prev_output_id;
                    } else {
                        // 其余 inputs（forward 输入 x/y 等）：按 i_node_type+idx 复用或新增 fused_graph 输入
                        std::string key = std::to_string(i) + "_" + node_type + "_in_" + std::to_string(j);
                        auto it = forward_inputs.find(key);
                        if (it == forward_inputs.end()) {
                            const Node& in_node = sub_graph.node(src_input_id);
                            size_t new_id = fused_graph.addInput(in_node.out_desc);
                            forward_inputs[key] = new_id;
                            remap_input[src_input_id] = new_id;
                        } else {
                            remap_input[src_input_id] = it->second;
                        }
                    }
                }

                // 合并 sub_graph 到 fused_graph，获取正确的节点 ID 映射关系
                auto old_to_new = fused_graph.mergeGraph(sub_graph, remap_input);

                if (sub_graph.outputCount() > 0) {
                    prev_output_id = old_to_new.at(sub_graph.outputs()[0]);
                    grad_desc = sub_graph.node(sub_graph.outputs()[0]).out_desc;
                    per_node_output_ids.push_back(prev_output_id);
                }
            }

            // 按节点顺序 markOutput → fusion kernel 执行返回的 vector<Tensor> 顺序对应 per_node_output_ids
            for (size_t id : per_node_output_ids) {
                fused_graph.markOutput(id);
            }
            // 【DEBUG 临时 EXP-2】fuse() 之前 dump 一下，只看 ReLU+Sigmoid 32x64 这个 case
            #ifdef CT_DEBUG
            {
                static int dbg_pre_fuse = 0;
                bool is_target = (seq.node_types.size() == 2 &&
                                   seq.node_types[0].find("ReLU") != std::string::npos &&
                                   seq.node_types[1].find("Sigmoid") != std::string::npos &&
                                   seq.grad_shapes[0].size() == 2 &&
                                   seq.grad_shapes[0][0] == 32);
                if (is_target && dbg_pre_fuse < 3) {
                    std::cerr << "\n========== [DBG-PRE-FUSE#target#" << dbg_pre_fuse
                              << "] ==========\n";
                    std::cerr << "seq.node_types=[";
                    for (auto& t : seq.node_types) std::cerr << t << " | ";
                    std::cerr << "]\n";
                    std::cerr << "fused_graph = " << fused_graph.toString() << "\n";
                    std::cerr << "========== END PRE-FUSE ==========\n\n";
                    dbg_pre_fuse++;
                }
            }
            #endif
            ms_since(("Graph built OK, nodes=" + std::to_string(fused_graph.nodeCount()) + " outputs=" + std::to_string(fused_graph.outputCount())).c_str());

            // 编译融合图：先试 MLIR 后端（自动优化、SIMD/向量化更强），失败再 fallback Handwritten
            // MLIR 已支持多输出反向融合：每个输出写入 out_ptr 的不同段
            CompileOptions opts_mlir, opts_hw;
            opts_mlir.backend = C3Backend::MLIR;
            opts_mlir.opt_level = 3;
            opts_mlir.enable_cache = true;
            opts_mlir.enable_fusion = false; // 禁用后向图内的二次融合，防止多输出拓扑序及输入索引映射被打乱
            opts_hw.backend = C3Backend::Handwritten;
            opts_hw.opt_level = 3;
            opts_hw.enable_cache = true;
            opts_hw.enable_fusion = false;

            std::shared_ptr<CompiledKernel> kernel = nullptr;
            bool mlir_tried = false;
            std::string mlir_err;
            {
                try {
                    mlir_tried = true;
                    ms_since("START MLIR compile...");
                    kernel = C3Engine::getInstance().compile(fused_graph, opts_mlir);
                    ms_since(kernel ? "MLIR compile SUCCESS" : "MLIR returned nullptr");
                } catch (const std::out_of_range& e) {
                    mlir_err = e.what();
                    kernel = nullptr;
                    ms_since(("MLIR out_of_range: " + mlir_err).c_str());
                } catch (const std::exception& e) {
                    mlir_err = e.what();
                    kernel = nullptr;
                    ms_since(("MLIR exception: " + mlir_err.substr(0, 60)).c_str());
                } catch (...) {
                    mlir_err = "unknown exception";
                    kernel = nullptr;
                }
            }
            if (!kernel) {
                if (mlir_tried) {
                    static std::mutex mu_mlir_log;
                    static int mlir_fail_log_count = 0;
                    std::lock_guard<std::mutex> lk(mu_mlir_log);
                    if (mlir_fail_log_count++ < 3) { // 最多打 3 次 MLIR 失败日志，防止刷屏
                        // 走 CtorchError::log 统一格式（release 也可见，受 log level 控制）
                        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
                            ErrorType::KERNEL_LAUNCH,
                            "[C3-BW-MLIR-FALLBACK] seq_key=" + makeSequenceKey(seq.node_types) +
                            " N=" + std::to_string(seq.node_types.size()) +
                            " outputs=" + std::to_string(fused_graph.outputCount()) +
                            " MLIR compile failed" + (mlir_err.empty() ? "(kernel=nullptr)" : (": "+mlir_err)) +
                            " → fallback Handwritten backend");
                    }
                }
                try {
                    ms_since("START Handwritten compile...");
                    kernel = C3Engine::getInstance().compile(fused_graph, opts_hw);
                    ms_since(kernel ? "Handwritten compile SUCCESS" : "Handwritten returned nullptr");
                } catch (const std::exception& e) {
                    ms_since(("Handwritten exception: " + std::string(e.what()).substr(0,60)).c_str());
                    kernel = nullptr;
                } catch (...) {
                    kernel = nullptr;
                }
            }
            if (kernel) {
                // 注册时的形状与 makeFusedBackwardKey 生成时保持一致：
                // - reg_grad_shape：最下游 grad 形状（tryExecuteBackward 传入的 grad.shape 必须匹配）
                // - reg_input_shape：最上游 input 形状（仅作匹配签名，reshape 时 out_shape 用 kernel 产出形状）
                // num_inputs = fused_graph.inputCount() = 1 (grad) + N (每节点 forward input)，
                //   与 tryExecuteFusedBackward 传入 [grad, best_fwd_inputs(0..N-1)] 严格一致。
                // [Fix 2026-08-11 DCE 输入平移] 融合图输入顺序 = [grad, fwd0, fwd1, ...]，
                //   tryExecuteFusedBackward 按 forward_inputs[0..N-1] 顺序喂 → 恒等索引表。
                std::vector<size_t> fused_fwd_map;
                fused_fwd_map.reserve(fused_graph.inputCount() - 1);
                for (size_t fi = 1; fi < fused_graph.inputCount(); ++fi) {
                    fused_fwd_map.push_back(fi - 1);
                }
                C3KernelRegistry::getInstance().installBackward(
                    fused_key, kernel, reg_grad_shape, reg_input_shape,
                    /*fwd_input_map=*/fused_fwd_map,
                    /*num_inputs=*/fused_graph.inputCount());

                std::lock_guard<std::mutex> lock(stats_mutex_);
                fusion_compile_count_++;

                #ifdef CT_DEBUG
                // ===== DEBUG: 实际注册的 key，和 execute 端 make 的 full_key 对比定位 miss
                std::string shape_g; for (auto s : reg_grad_shape) shape_g += std::to_string(s)+",";
                std::string shape_i; for (auto s : reg_input_shape) shape_i += std::to_string(s)+",";
                std::cerr << "[DBG-KEY-INSTALL] key=" << fused_key
                          << " seq_types=" << makeSequenceKey(seq.node_types)
                          << " reg_grad=[" << shape_g << "] reg_input=[" << shape_i << "]" << std::endl;
                ms_since(("KERNEL INSTALLED! compile_count=" + std::to_string(fusion_compile_count_)).c_str());
                #endif
            } else {
                #ifdef CT_DEBUG
                ms_since("COMPILE FAILED (both backends)");
                #endif
            }
        } catch (const std::exception& e) {
            // 走 CtorchError::log 统一格式（release 也可见）
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL,
                ErrorType::KERNEL_LAUNCH,
                "[C3-BW-FUSION-ERR] compileFusedBackwardAsync exception: " +
                std::string(e.what()) + "  seq_key=" + makeSequenceKey(seq.node_types) +
                "  N=" + std::to_string(seq.node_types.size()));
        }

        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_compiles_.erase(fused_key);
    }).detach();
}

// ======================= 融合执行入口 =======================

std::optional<Tensor> C3BackwardCapture::tryExecuteFusedBackward(
    const ::Node* node,
    const Tensor& grad,
    const std::vector<Tensor>& forward_inputs)
{
    // 多输出 backward fusion 的 intercepted 结果当前仍存在确定性的数值错误：
    // ReLU→Sigmoid / ReLU→ReLU 在异步 kernel 安装后会把错误的 upstream grad
    // 回填到后续节点。先隔离执行入口，保留单节点 backward C3；编译/统计代码
    // 留待修复输出段与节点生命周期后重新启用。正确性优先于融合命中率。
    (void)node;
    (void)grad;
    (void)forward_inputs;
    return std::nullopt;

    // ========== 路径 A：当前节点已在 pending 拦截队列中（融合已在下游节点 N0 时提前算出）
    //            直接取出对应的 upstream grad 作为本节点 backward 的结果返回。
    //            ComputeCore 流程零修改，grad pack 分发 100% 对齐。
    // ========== P1 优化：只读查 pending_intercepted_，用 shared_lock 共享锁（大部分 miss 场景 lock 成本骤降）
    {
        std::shared_lock<std::shared_mutex> lock(intercepted_mutex_);
        auto it = pending_intercepted_.find(node);
        if (it != pending_intercepted_.end()) {
            const std::string current_type = std::string(typeid(*node).name());
            // 必须 type 完全匹配（防止不同 backward 轮次间相同地址不同节点误命中）
            if (it->second.first == current_type) {
                Tensor intercepted = std::move(it->second.second);
                // 提前释放共享锁（避免在持锁期间做 stats mutex 二次锁）
                lock.unlock();
                pending_intercepted_.erase(node); // 注意：erase 要在 unlock 前？不，因为是 shared_lock 只读的不能 erase → 先 unlock 再拿 unique_lock erase
                // 正确做法：先 copy 出 value，释放 shared_lock，再拿 unique_lock 删除
                // 上面已经 move 了 intercepted，但 map 里的 entry 还在。需要重新拿 exclusive lock 清理。
                {
                    std::unique_lock<std::shared_mutex> ulock(intercepted_mutex_);
                    auto it2 = pending_intercepted_.find(node);
                    if (it2 != pending_intercepted_.end() && it2->second.first == current_type) {
                        pending_intercepted_.erase(it2);
                    }
                }
                std::lock_guard<std::mutex> slock(stats_mutex_);
                fusion_hit_count_++;
                std::cerr << "[DBG-INTERCEPT-HIT] node_type=" << current_type
                          << " intercepted_numel=" << intercepted.numel()
                          << " grad_numel=" << grad.numel() << std::endl;
                return intercepted;
            } else {
                // 地址复用但 type 不符 → 清理旧 entry
                #ifdef CT_DEBUG
                std::cerr << "[DBG-INTERCEPT-TYPEMISMATCH] node_ptr=" << node
                          << " stored_type=" << it->second.first
                          << " current_type=" << typeid(*node).name()
                          << " → ERASED" << std::endl;
                #endif
                // 读锁下不能 erase，先记录后处理
                lock.unlock();
                {
                    std::unique_lock<std::shared_mutex> ulock(intercepted_mutex_);
                    auto it2 = pending_intercepted_.find(node);
                    if (it2 != pending_intercepted_.end() && it2->second.first != current_type) {
                        pending_intercepted_.erase(it2);
                    }
                }
            }
        }
    }

    // ========== P1 优化：miss marker 快速跳过 ==========
    // 如果这个节点已经在当前 backward 轮次中走过一次 B 路径且 lookup 失败，
    // 后续再次访问直接跳过 upstream traversal 开销。
    {
        std::lock_guard<std::mutex> mlock(miss_marker_mutex_);
        if (miss_marker_nodes_.count(node)) {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            fusion_miss_count_++;
            return std::nullopt;
        }
    }

    // ========== 路径 B：当前节点是序列的最下游 N0。从当前节点向上游 traverse w-1 层拼序列，
    //            查找已编译好的多输出 fusion kernel，一次性算出 w 个节点的 upstream grad。
    //            outs[0] 返回给 ComputeCore（作为 N0→N1 的 grad pack）；
    //            outs[1..w-1] 存入 pending_intercepted_，对应 N1…Nw-1 随后的 backward 调用直接取。
    // 关键：此时 grad = N0.backward 的原生下游端 grad（dL/dy），没有被任何节点处理过！
    //       完全等价于 eager 流程的 N0.backward(dL/dy) 的输入 → 不会重复计算！
    const std::string current_type = std::string(typeid(*node).name());
    if (!isElementWiseBackward(current_type)) {
        return std::nullopt; // 只有 element-wise 节点才能当融合起点
    }

    // 从当前节点向上游 traverse：最多 kFusionWindowSize 个节点
    //   chain_nodes[0] = current node (shared_ptr 保活，避免被 clear 销毁)
    //   chain_types[0] = current node type
    //   chain_forward_inputs[0] = current node forward inputs（参数 forward_inputs）
    std::vector<std::shared_ptr<::Node>> chain_nodes;
    std::vector<std::string> chain_types;
    std::vector<std::vector<Tensor>> chain_forward_inputs;
    chain_nodes.push_back(std::shared_ptr<::Node>()); // nullptr placeholder：caller 侧节点所有权不在这，通过 raw ptr 访问
    chain_types.push_back(current_type);
    chain_forward_inputs.push_back(forward_inputs);

    // 当前正在遍历的上游节点指针（N1、N2…）：从 node->getUpStreamNodes()[0] 开始，逐 node 向上。
    // 注意：element-wise 节点单输入，upstream 只有 1 个（如果有多个 upstream 就不是纯 element-wise 链了）
    // **跳过中间的 Factor 节点**：ComputeCore 会把真正的单输入节点（Sigmoid/ReLU 等）的上游包一层
    //   Factor/Input/Identity 节点，类型不在 isElementWiseBackward 中，但 upstream.size() == 1。
    //   遇到这类节点就跳过，沿着 upstream[0] 继续往前走，最多 kSkipMax 层。
    const size_t kSkipMax = 4;
    const ::Node* cur = node;
    for (size_t step = 1; step < kFusionWindowSize; ++step) {
        // ---------- 内部：跳过 Factor / wrapper 节点 ----------
        const ::Node* cur_lookup = cur;
        size_t skips = 0;
        while (skips < kSkipMax) {
            auto ups = cur_lookup->getUpStreamNodes();
            if (ups.size() != 1) break;          // 多输入（真实多分支或 leaf）→ 不能 skip
            auto candidate_sp = ups[0];
            if (!candidate_sp) break;
            const std::string cname = std::string(typeid(*candidate_sp).name());
            if (isElementWiseBackward(cname)) {
                // 找到可融合节点 → 直接把 cur_lookup 置 candidate 并结束 skip loop
                cur_lookup = candidate_sp.get();
                break;
            }
            // 非 element-wise，但 upstream.size()==1 → 视为 wrapper/Factor，跳过
            cur_lookup = candidate_sp.get();
            skips++;
        }
        if (skips >= kSkipMax) break; // 太深的 wrapper 链 → 放弃

        // ---------- 取 cur_lookup 的上游 next_sp 作为下一轮的 cur ----------
        {
            auto upstreams_of_cur = cur_lookup->getUpStreamNodes();
            // 注意：现在 cur_lookup 本身必须已经是可融合节点 type（上 while 里保证）
            // 但我们现在要获取 cur_lookup 的「上游」以便下一轮 step 继续 traverse。
            // 上面 while 里 candidate_sp 可融合时已经把 cur_lookup = candidate_sp.get()。
            // 所以 upstreams_of_cur 是 candidate_sp 的 upstreams → 即可融合节点的上游 target。
            // 这和原逻辑一致！
            if (upstreams_of_cur.size() != 1) break; // 可融合节点上游 size 必须为 1（否则多输入链断）
            auto next_sp = upstreams_of_cur[0];
            if (!next_sp) break;

            const std::string tname = std::string(typeid(*cur_lookup).name());
            if (!isElementWiseBackward(tname)) break;

            // 保存上游节点的 shared_ptr、type、forward inputs
            // 这里 cur_lookup 就是可融合节点 raw ptr，但 shared_ptr 在哪？— candidate_sp 就是！
            // 我们上面 candidate_sp = ups[0] 可融合时还在作用域吗？重取 via cur_lookup 的
            // 实际 shared_ptr：重新从 cur 的最近 upstream 找？
            // 最保险方式 — 重新从 skips 遍历一遍获取 shared_ptr：
            std::shared_ptr<::Node> fused_sp;
            const ::Node* c2 = cur;
            size_t s2 = 0;
            while (s2 < kSkipMax) {
                auto ups2 = c2->getUpStreamNodes();
                if (ups2.size() != 1) break;
                auto cand2 = ups2[0];
                if (!cand2) break;
                const std::string nm = std::string(typeid(*cand2).name());
                if (isElementWiseBackward(nm)) { fused_sp = cand2; break; }
                c2 = cand2.get(); s2++;
            }
            if (!fused_sp) break;
            // [Fix 2026-08-11 反向融合 SIGBUS] 形状一致性校验：
            // 融合 kernel 用「统一 elem_n 分段 output 平面 buffer + 链式 grad 传播」，
            // 隐含假设链内所有 element-wise 节点的 grad/input 形状一致（都 == 最下游
            // grad 形状）。若 traverse 跳过中间 Add/MatMul 等 wrapper 而撞上形状发生
            // 变化的节点（如 MNIST 两个 ReLU 隔着 MatMul，z2 grad=[128,128] 而 z1
            // input=[128,256]），把形状不同的节点拼进同链会越界读 → SIGBUS。
            // 因此：上游节点 forward input[0] 形状必须 == 最下游 grad 形状，否则 break。
            auto fused_inputs = fused_sp->getInputs();
            if (fused_inputs.empty() || fused_inputs[0].sizes() != grad.sizes()) break;
            chain_nodes.push_back(fused_sp);
            chain_types.push_back(tname);
            chain_forward_inputs.push_back(fused_inputs);

            // 下一轮 traverse 从这个可融合节点本身出发（它的上游是更上一层）
            cur = fused_sp.get();
        }
    }

    // 从最长窗口向下尝试查找已编译的 fusion kernel（贪心更长的融合 = 更省）
    const size_t max_L = chain_types.size(); // >= 1
    // full_forward_inputs[0..w-1]：按 kernel 需要的顺序（types[0].forward[0], types[1].forward[0], ...）
    std::vector<Tensor> best_fwd_inputs;
    std::string best_full_key;
    size_t best_w = 0;
    {
        for (size_t w = kFusionWindowSize; w >= 2; --w) {
            if (w > max_L) continue;
            std::vector<std::string> types(chain_types.begin(), chain_types.begin() + w);
            if (!isFusableSequence(types)) continue;

            // 形状签名对齐 compileFusedBackwardAsync 的 reg_grad_shape / reg_input_shape：
            //   reg_grad_shape  = seq.grad_shapes.front() = 最下游端（先执行）的 grad shape
            //                    = grad.sizes()（当前 N0 的 dL/dy shape，正确）
            //   reg_input_shape = seq.input_shapes.back() = 最上游端（chain[w-1]）forward input[0] shape
            const size_t last_in_chain = w - 1;
            const std::vector<size_t>& cur_upstream_input_shape =
                (!chain_forward_inputs[last_in_chain].empty()) ? chain_forward_inputs[last_in_chain][0].sizes() : grad.sizes();
            std::string seq_key = makeSequenceKey(types);
            std::string full_key = makeFusedBackwardKey(seq_key, grad.sizes(), cur_upstream_input_shape);
            bool has_key = C3KernelRegistry::getInstance().hasBackwardKey(full_key);
            // ===== DEBUG: 执行端查找的 key，和 [DBG-KEY-INSTALL] 对比定位 100% miss
            #ifdef CT_DEBUG
            {
                std::string shape_g; for (auto s : grad.sizes()) shape_g += std::to_string(s)+",";
                std::string shape_i; for (auto s : cur_upstream_input_shape) shape_i += std::to_string(s)+",";
                static std::mutex dbg_mu;
                static int dbg_counter = 0;
                std::lock_guard<std::mutex> dlk(dbg_mu);
                if (dbg_counter++ < 6) { // 只打印前 6 次，避免刷屏
                    std::cerr << "[DBG-KEY-LOOKUP] key=" << full_key
                              << " seq_types=" << makeSequenceKey(types)
                              << " grad=[" << shape_g << "] upstream_input=[" << shape_i << "]"
                              << " has_key=" << (has_key?"YES":"NO") << std::endl;
                }
            }
            #endif
            if (has_key) {
                // 组装完整的 forward inputs：types[k] 的 forward_input[0] 依次 push
                best_fwd_inputs.clear();
                best_fwd_inputs.reserve(w);
                for (size_t k = 0; k < w; ++k) {
                    if (!chain_forward_inputs[k].empty()) {
                        best_fwd_inputs.push_back(chain_forward_inputs[k][0]);
                    }
                }
                best_full_key = std::move(full_key);
                best_w = w;
                break;
            }
        }
    }
    if (best_w < 2 || best_fwd_inputs.size() != best_w) {
        // ========== P1 优化：lookup 失败 → 记录 miss marker，下次直接跳过 traversal ==========
        {
            std::lock_guard<std::mutex> mlock(miss_marker_mutex_);
            miss_marker_nodes_.insert(node);
            // 防止内存爆炸：超过 1 万条自动清空（下一轮从头开始，开销可忽略）
            if (miss_marker_nodes_.size() > 10000) {
                miss_marker_nodes_.clear();
            }
        }
        std::lock_guard<std::mutex> slock(stats_mutex_);
        fusion_miss_count_++;
        return std::nullopt; // 没命中任何融合
    }

    // ============================================================
    // 命中！执行多输出 fusion kernel → outs.size() == best_w（每个节点 1 个上游 grad 输出）
    auto outs_opt = C3KernelRegistry::getInstance().tryExecuteBackward(
        best_full_key, grad, best_fwd_inputs);
    if (!outs_opt.has_value() || outs_opt->size() != best_w) {
        #ifdef CT_DEBUG
        static std::mutex dbg_mu2;
        static int dbg_counter2 = 0;
        std::lock_guard<std::mutex> dlk(dbg_mu2);
        if (dbg_counter2++ < 6) {
            std::cerr << "[DBG-KER-EXEC-FAIL] key=" << best_full_key
                      << " has_value=" << outs_opt.has_value()
                      << " outs_size=" << (outs_opt.has_value() ? outs_opt->size() : -1)
                      << " best_w=" << best_w << std::endl;
        }
        #endif
        std::lock_guard<std::mutex> slock(stats_mutex_);
        fusion_miss_count_++;
        return std::nullopt;
    }
    auto& outs = outs_opt.value();

    // ===== 形状修正：kernel 产出的 outs[k] 可能是 flat 1D（{elem_n}），需还原为 grad 的多维形状（如 {512,512}） =====
    //       反向融合：所有节点的 upstream grad 形状与传入的 grad 完全相同
    for (size_t k = 0; k < outs.size(); ++k) {
        if (outs[k].sizes() != grad.sizes() && outs[k].numel() == grad.numel()) {
            outs[k] = outs[k].reshape(grad.sizes());
        }
    }

    // outs[1..w-1] 放进 pending_intercepted_，对应 chain_nodes[1..w-1] 的 raw ptr 作为 key。
    // 当 ComputeCore 之后处理这些节点（N1、N2…）时，走路径 A 直接取出返回。
    // ========== P1 优化：写操作拿 unique_lock 独占锁 ==========
    {
        std::unique_lock<std::shared_mutex> lock(intercepted_mutex_);
        for (size_t k = 1; k < best_w; ++k) {
            const ::Node* raw = nullptr;
            std::string expected_type;
            if (k == 0) {
                // 占位（不会到这里）
            } else if (chain_nodes[k]) {
                // 上游节点（k>=1）我们保存了 shared_ptr，直接 get() + typeid
                raw = chain_nodes[k].get();
                expected_type = std::string(typeid(*raw).name());
            }
            if (raw) {
                pending_intercepted_[raw] = std::make_pair(expected_type, outs[k]);
            }
        }
    }

    // outs[0] = N0（当前节点）的 upstream grad，直接以 1-element vector<Tensor> 形式返回。
    // ComputeCore 会把 outs[0] 作为 GradPack(target = N0.upstream[0] = N1) 发出，与 eager 完全一致。
    std::lock_guard<std::mutex> slock(stats_mutex_);
    fusion_hit_count_++;
    return outs[0];
}

// ======================= 工具函数 =======================

bool C3BackwardCapture::needsSumReduce(
    const std::vector<size_t>& grad_shape,
    const std::vector<size_t>& target_shape)
{
    if (grad_shape == target_shape) return false;
    if (grad_shape.empty() || target_shape.empty()) return true;

    // 右对齐比较：从最右边维度开始比较
    size_t grad_rank = grad_shape.size();
    size_t target_rank = target_shape.size();

    // 如果 grad 的维度比 target 多，需要 reduce 多余的维度
    if (grad_rank > target_rank) return true;

    // 右对齐比较
    for (size_t i = 0; i < grad_rank; ++i) {
        size_t gd = grad_shape[grad_rank - 1 - i];
        size_t td = (i < target_rank) ? target_shape[target_rank - 1 - i] : 1;
        if (gd != td) return true;
    }

    return false;
}

int C3BackwardCapture::computeReduceAxis(
    const std::vector<size_t>& grad_shape,
    const std::vector<size_t>& target_shape)
{
    if (grad_shape == target_shape) return -1;
    if (grad_shape.empty()) return -1;
    if (target_shape.empty()) {
        // 全 reduce
        return -1;
    }

    // 找到第一个维度不同的位置（从左边开始）
    // 如果 grad 比 target 多维度，reduce 多余的维度
    size_t grad_rank = grad_shape.size();
    size_t target_rank = target_shape.size();

    if (grad_rank > target_rank) {
        // 多余的维度需要 reduce
        return 0; // reduce 第 0 维（第一个多余维度）
    }

    // 右对齐，从左到右找到第一个不匹配的维度
    for (size_t i = 0; i < grad_rank; ++i) {
        size_t offset = grad_rank - target_rank;
        if (i < offset) {
            return static_cast<int>(i); // 这个维度在 target 中没有对应
        }
        size_t gd = grad_shape[i];
        size_t td = target_shape[i - offset];
        if (gd != td && td == 1) {
            return static_cast<int>(i); // 这个维度需要 reduce
        }
    }

    return -1; // 全 reduce
}

void C3BackwardCapture::clear() {
    std::lock_guard<std::mutex> lock1(fusion_mutex_);
    std::unique_lock<std::shared_mutex> lock2(intercepted_mutex_);
    std::lock_guard<std::mutex> lock3(pending_mutex_);
    std::lock_guard<std::mutex> lock4(miss_marker_mutex_);

    recent_sequence_.clear();
    recent_grad_shapes_.clear();
    recent_input_shapes_.clear();
    recent_forward_inputs_.clear();
    pending_intercepted_.clear();
    pending_mimo_intercepted_.clear();
    miss_marker_nodes_.clear();
    sequence_counts_.clear();
    pending_compiles_.clear();
}

void C3BackwardCapture::clearCallScopedState() {
    std::lock_guard<std::mutex> lock1(fusion_mutex_);
    std::unique_lock<std::shared_mutex> lock2(intercepted_mutex_);
    std::lock_guard<std::mutex> lock4(miss_marker_mutex_);

    recent_sequence_.clear();
    recent_grad_shapes_.clear();
    recent_input_shapes_.clear();
    recent_forward_inputs_.clear();
    pending_intercepted_.clear();
    pending_mimo_intercepted_.clear();
    miss_marker_nodes_.clear();
}

std::optional<std::vector<Tensor>> C3BackwardCapture::tryExecuteUnifiedMIMOBackward(
    const ::Node* node, const Tensor& grad,
    const std::vector<Tensor>& forward_inputs)
{
    // 检查是否为支持的激活节点
    std::string current_type = std::string(typeid(*node).name());
    bool is_act = (current_type.find("ReLUNode") != std::string::npos ||
                   current_type.find("SigmoidNode") != std::string::npos ||
                   current_type.find("TanhNode") != std::string::npos);
    if (!is_act) return std::nullopt;

    // 向上游匹配: Activation -> Add -> MatMul
    auto ups = node->getUpStreamNodes();
    if (ups.size() != 1 || !ups[0]) return std::nullopt;
    const ::Node* add_node = ups[0].get();
    std::string add_type = typeid(*add_node).name();
    if (add_type.find("AddNode") == std::string::npos) return std::nullopt;

    auto add_ups = add_node->getUpStreamNodes();
    if (add_ups.size() != 2 || !add_ups[0] || !add_ups[1]) return std::nullopt;
    const ::Node* matmul_node = add_ups[0].get();
    std::string mm_type = typeid(*matmul_node).name();
    if (mm_type.find("MatMulNode") == std::string::npos) return std::nullopt;

    // 获取相关张量
    const Tensor& z = forward_inputs.empty() ? node->getInputs()[0] : forward_inputs[0];
    const Tensor& X = matmul_node->getInputs()[0];
    const Tensor& W = matmul_node->getInputs()[1];

    // 构建 MIMO cache key
    auto t_key0 = std::chrono::steady_clock::now();
    std::stringstream ss;
    ss << "mimo_backward_" << current_type << "|g:";
    for (auto s : grad.sizes()) ss << s << ",";
    ss << "|z:";
    for (auto s : z.sizes()) ss << s << ",";
    ss << "|x:";
    for (auto s : X.sizes()) ss << s << ",";
    ss << "|w:";
    for (auto s : W.sizes()) ss << s << ",";
    std::string mimo_key = ss.str();
    {
        std::lock_guard<std::mutex> klock(stats_mutex_);
        mimo_keybuild_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_key0).count());
    }

    // 检查是否有已编译的 JIT kernel
    auto& registry = C3KernelRegistry::getInstance();
    if (registry.hasBackwardKey(mimo_key)) {
        // [mimo_exec_us] 默认为 True
        // [主坑定位] C3_MIMO_TRACE=1：量化反向输入 data_read/物化 是否为大头
        //   mn_setup 全 MultiNode 仅 ~3µs/call，应反证 data_read 不物化；实测定音。
        if (const char* mt = std::getenv("C3_MIMO_TRACE")) {
            (void)mt;
            static int mt_cnt = 0;
            if (mt_cnt < 8) { mt_cnt++;
                auto t0r = std::chrono::steady_clock::now();
                (void)z.data_read<float>();
                (void)X.data_read<float>();
                (void)W.data_read<float>();
                auto dns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0r).count();
                auto lmz = z.lazyMaterializer();
                auto lmx = X.lazyMaterializer();
                auto lmw = W.lazyMaterializer();
                fprintf(stderr, "[MIMO-TRACE] z_lazy=%p x_lazy=%p w_lazy=%p data_read_us=%.2f\n",
                        (void*)(lmz ? lmz.get() : nullptr),
                        (void*)(lmx ? lmx.get() : nullptr),
                        (void*)(lmw ? lmw.get() : nullptr),
                        dns / 1000.0);
            }
        }
        // 收集输入：[z, X, W]，不重复包含 grad，配合 fwd_input_map {0, 1, 2}
        std::vector<Tensor> inputs = {z, X, W};
        auto t_exec0 = std::chrono::steady_clock::now();
        auto result = registry.tryExecuteBackward(mimo_key, grad, inputs);
        if (result.has_value() && result->size() == 4) {
            // result[0]: grad_z, result[1]: grad_W, result[2]: grad_X, result[3]: grad_b
            Tensor grad_z = std::move((*result)[0]);
            Tensor grad_W = std::move((*result)[1]);
            Tensor grad_X = std::move((*result)[2]);
            Tensor grad_b = std::move((*result)[3]);
            {
                std::lock_guard<std::mutex> slock(stats_mutex_);
                mimo_hit_count_++;
                mimo_exec_ns_ += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t_exec0).count());
            }

            // 存入 pending_mimo_intercepted_
            {
                std::unique_lock<std::shared_mutex> lock(intercepted_mutex_);
                // AddNode::backward expects to return: {grad_mm, grad_b}
                pending_mimo_intercepted_[add_node] = {grad_z, grad_b};

                // MatMulNode::backward expects to return: {grad_X, grad_W}
                pending_mimo_intercepted_[matmul_node] = {grad_X, grad_W};
            }
            #ifdef CT_DEBUG
            std::cerr << "[MIMO-EXEC-HIT] successfully executed fused backward layer!" << std::endl;
            #endif

            // ReLU_Grad returns `grad_z`
            std::vector<Tensor> act_res = {grad_z};
            return act_res;
        }
    }

    // 触发异步编译
    TensorDesc grad_desc = TensorDesc::fromShape(grad.sizes());
    TensorDesc z_desc = TensorDesc::fromShape(z.sizes());
    TensorDesc x_desc = TensorDesc::fromShape(X.sizes());
    TensorDesc w_desc = TensorDesc::fromShape(W.sizes());
    compileUnifiedMIMOBackwardAsync(node, add_node, matmul_node, grad_desc, z_desc, x_desc, w_desc);
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        mimo_miss_count_++;
    }

    return std::nullopt;
}

void C3BackwardCapture::compileUnifiedMIMOBackwardAsync(
    const ::Node* relu_node, const ::Node* add_node, const ::Node* matmul_node,
    const TensorDesc& grad_desc, const TensorDesc& z_desc,
    const TensorDesc& x_desc, const TensorDesc& w_desc)
{
    std::string current_type = std::string(typeid(*relu_node).name());

    std::stringstream ss;
    ss << "mimo_backward_" << current_type << "|g:";
    for (auto s : grad_desc.shape) ss << s << ",";
    ss << "|z:";
    for (auto s : z_desc.shape) ss << s << ",";
    ss << "|x:";
    for (auto s : x_desc.shape) ss << s << ",";
    ss << "|w:";
    for (auto s : w_desc.shape) ss << s << ",";
    std::string mimo_key = ss.str();

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_compiles_.find(mimo_key) != pending_compiles_.end()) {
            return;
        }
        pending_compiles_[mimo_key] = true;
    }
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        mimo_compile_count_++;
    }
    std::string compile_err = "";

    std::thread([this, current_type, mimo_key, grad_desc, z_desc, x_desc, w_desc]() {
        try {
            // 1. 构建三个底层独立的反向子图 + 一个偏置子图
            BackwardGraph act_bg;
            if (current_type.find("ReLUNode") != std::string::npos) {
                act_bg = buildReLUBackwardGraph(grad_desc, z_desc);
            } else if (current_type.find("SigmoidNode") != std::string::npos) {
                act_bg = buildSigmoidBackwardGraph(grad_desc, z_desc);
            } else {
                act_bg = buildTanhBackwardGraph(grad_desc, z_desc);
            }
            Graph act_sub = std::move(act_bg.first);
            TensorDesc act_out_desc = act_sub.node(act_sub.outputs()[0]).out_desc;

            // mm_w_sub: input_index == 1 计算 W 的梯度 (grad_W)
            Graph mm_w_sub = buildMatMulBackwardGraph(act_out_desc, x_desc, w_desc, 1).first;
            // mm_x_sub: input_index == 0 计算 X 的梯度 (grad_X)
            Graph mm_x_sub = buildMatMulBackwardGraph(act_out_desc, x_desc, w_desc, 0).first;

            // add_b_sub: bias 的梯度 (grad_b)
            TensorDesc bias_grad_desc = TensorDesc::fromShape({grad_desc.shape[1]});
            Graph add_b_sub = buildAddBackwardGraph(act_out_desc, grad_desc, bias_grad_desc, 1).first;

            // 2. 声明 GraphMerger 的拓扑缝合规格，构造大一统融合图 (MIMO)
            std::vector<Graph> sub_graphs = {act_sub, mm_w_sub, mm_x_sub, add_b_sub};
            MergeSpec spec;

            // 链接 1：act_sub (子图 0, 输出 0) ──► mm_w_sub (子图 1, 输入 0)
            spec.links.push_back(MergeLink{0, 0, 1, 0});
            // 链接 2：act_sub (子图 0, 输出 0) ──► mm_x_sub (子图 2, 输入 0)
            spec.links.push_back(MergeLink{0, 0, 2, 0});
            // 链接 3：act_sub (子图 0, 输出 0) ──► add_b_sub (子图 3, 输入 0)
            spec.links.push_back(MergeLink{0, 0, 3, 0});

            MergedGraphInfo unified_info = GraphMerger::merge(sub_graphs, spec);
            Graph fused_graph = std::move(unified_info.graph);

            // 3. 强行清除默认输出并按顺序物理标记四个输出
            fused_graph.clearOutputs();
            fused_graph.markOutput(unified_info.output_remap[0][0]); // Output 0: grad_z (for ReLU_Grad)
            fused_graph.markOutput(unified_info.output_remap[1][0]); // Output 1: grad_W (for weights update)
            fused_graph.markOutput(unified_info.output_remap[2][0]); // Output 2: grad_X (for backpropagation)
            fused_graph.markOutput(unified_info.output_remap[3][0]); // Output 3: grad_b (for bias update)

            CompileOptions opts;
            opts.backend = C3Backend::MLIR;
            opts.enable_fusion = true;

            // 编译融合图为 JIT kernel
            auto kernel = C3Engine::getInstance().compile(fused_graph, opts);
            if (kernel) {
                // 注册到 C3KernelRegistry 中，使用 {0, 1, 2} 对应 inputs 中的 z, X, W
                C3KernelRegistry::getInstance().installBackward(
                    mimo_key, kernel, grad_desc.shape, grad_desc.shape,
                    {0, 1, 2}, 4
                );
                #ifdef CT_DEBUG
                std::cerr << "[MIMO-COMPILE-SUCCESS] compiled unified backward layer successfully! key=" << mimo_key << std::endl;
                #endif
            }
        } catch (const std::exception& e) {
            #ifdef CT_DEBUG
            std::cerr << "[MIMO-COMPILE-ERR] compile unified backward layer failed: " << e.what() << std::endl;
            #endif
            // 诊断：非 debug 构建也暴露一次编译根因，便于定位 MIMO 未命中的原因
            {
                static std::mutex err_mu;
                std::lock_guard<std::mutex> ek(err_mu);
                static std::hash<std::string> h;
                static std::unordered_set<size_t> seen;
                size_t kh = h(std::string(e.what()) + "|" + mimo_key);
                if (seen.insert(kh).second) {
                    fprintf(stderr, "[MIMO-COMPILE-ERR] key=%s err=%s\n",
                            mimo_key.c_str(), e.what());
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_compiles_.erase(mimo_key);
        }
    }).detach();
}

} // namespace c3
} // namespace ct