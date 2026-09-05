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
    // 由 shutdownAll() 显式停止 detached 任务；保留对象至进程结束，
    // 避免其 mutex/cv 在其他 TU 的静态析构阶段失效。
    static C3BackwardCapture* instance = new C3BackwardCapture();
    return *instance;
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

    // [BW-SEG2 2026-09-02] C3_BW_SEG2=1：对 tryExecuteBackward 整体分段，
    //   确认「非内核编排」overhead 到底落在哪一步（区别于 MIMO-SEG 管 MIMO 内部、
    //   PHASE1-MISS 管 miss）。
    //   分段语义（每段的是「该步骤耗时」，用相邻 mark 差分累加）：
    //     prefix  : disabled/开关检查 + MIMO pending 拦截 mutex 查找
    //     mimo    : tryExecuteUnifiedMIMOBackward 整体调用（含 MIMO-SEG 内部）
    //     phase2  : tryExecuteFusedBackward（单输入融合反向）
    //     phase1  : Phase1 逐输入 kernel key 构建 + 查找 + 执行循环
    //     miss    : 任一输入 miss → compileBackwardAsyncForInput + wait 同步等待
    //     wrap    : 结果组装 + 统计 + 返回
    static const bool bw_seg2 = [] {
        const char* e = std::getenv("C3_BW_SEG2");
        return e && std::string(e) == "1";
    }();
    static struct {
        std::atomic<uint64_t> n[7], c[7];
    } beseg;
    auto beprev = std::chrono::steady_clock::now();
    auto bemark = [&beprev](int seg) {
        if (!bw_seg2) return;
        auto now = std::chrono::steady_clock::now();
        uint64_t d = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - beprev).count();
        beseg.n[seg] += d; beseg.c[seg]++;
        beprev = now;
        static thread_local size_t beacc = 0;
        if ((++beacc) % 400 == 0) {
            const char* nm[7] = {"prefix","mimo","phase2","phase1","miss","wrap","other"};
            fprintf(stderr, "[BW-SEG2]");
            for (int i = 0; i < 7; ++i)
                fprintf(stderr, " %s=%.2fms/%llu", nm[i], beseg.n[i]*1e-3, (unsigned long long)beseg.c[i]);
            fprintf(stderr, "\n");
        }
    };
    // backward C3 默认启用；保留 C3_ENABLE_BACKWARD=0 作为显式关闭方式。
    // CTORCH_DISABLE_C3_BACKWARD=1 或 C3_DISABLE_BACKWARD=1 仍由上方统一开关处理。
    const char* enable_backward = std::getenv("C3_ENABLE_BACKWARD");
    if (enable_backward && std::string(enable_backward) == "0") {
        return std::nullopt;
    }

    // ===== 统一 MIMO 融合反向 (梯度传导 + 激活求导 + 权重收缩) 🌟 =====
    // [2026-08-31 恢复] 去掉 C3_ENABLE_MIMO_BACKWARD opt-in 门控，MIMO 默认开启
    //   （历史黄金态默认开：mimo_compile=2, hit=4678/epoch, MIMO bwd ~57ms/ep）。
    //   保留 f8161c6 的正确性守卫（Gt SelectOp / pool buffer 独立槽位 /
    //   Softmax·Sigmoid 回退 eager / tryExecuteFusedBackward 短路）。
    {
        std::unique_lock<std::shared_mutex> lock(intercepted_mutex_);
        auto it = pending_mimo_intercepted_.find(node);
        if (it != pending_mimo_intercepted_.end()) {
            auto grads = std::move(it->second);
            pending_mimo_intercepted_.erase(it);
            return grads;
        }
    }
    bemark(0); // prefix 段结束 → 进入 mimo 段
    auto mimo_res = tryExecuteUnifiedMIMOBackward(node, grad, forward_inputs);
    bemark(1); // mimo 段结束（无论 hit/miss，此时间点后进入后续逻辑或返回）
    if (mimo_res.has_value()) {
        return mimo_res;
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

    // [BW-FIX 2026-09-02] CrossEntropy 是末端双输入节点，其 eager backward
    //   (CrossEntropyNode::backward) 只给 logits(upstream[0]) 投 1 个梯度、**不给 target 投梯度**。
    //   C3 逐输入协议却强制 out.size()==fwd_inputs.size()（含 target）且逐输入 shape 匹配，
    //   两者语义冲突：强行让 in:0 命中并在 in:1 补零梯度，会把错误形状的梯度投给 target 上游，
    //   导致 GradBucket::add 的 Add_SIMD 形状不兼容崩溃（已实测）。故 CE 反向应始终静默走 eager，
    //   在此**直接短路 return nullopt**，彻底摘除 CE 的逐输入查找/execute/miss 链路——既符合
    //   语义，也消除了 CE 历史上每 batch ~399 次「已装表 key 但仍 execute 失败」的表观 miss。
    if (type_name.find("CrossEntropyNode") != std::string::npos) {
        return std::nullopt;
    }

    if (n_inputs == 1) {
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

    bemark(2); // phase2 段结束 → 进入 phase1 段
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
        // [BW-FIX 2026-09-02] Add 反向「无广播侧」是恒等 passthrough：当目标输入形状 ==
        // grad 形状（如 bias 加的激活侧 in:0，grad:128,256 → dx0=grad），完全不需要 kernel。
        // 历史 bug：buildAddBackwardGraph 对它产出「无算力节点」图 → worker 在
        // nodeCount<=inputCount 处跳过、从不装表 → 每批都重起线程 + waitForPendingCompiles
        // 忙轮询白等 ~646µs（约占 backward 85%）。这里直接返回 grad，砍掉整条链路。
        if (type_name.find("AddNode") != std::string::npos) {
            const auto& in0 =
                (!forward_inputs.empty()) ? forward_inputs[i] : node->getInputs()[i];
            if (in0.sizes() == grad.sizes()) {
                out.push_back(grad);
                continue; // 该输入无需 kernel；其余输入（如广播 bias 侧）继续正常编译
            }
        }
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
                // [BW-MISS 2026-09-02] 从「仅打印首次 key」升级为「按 key 统计累计 miss 次数」，
                //   定期输出 top 来源，以区分「首轮 miss 后即命中」vs「每 batch 稳定 miss」。
                static std::mutex mt_mu;
                static std::unordered_map<std::string, uint64_t> mt_count;
                static uint64_t mt_total = 0;
                std::string mkey = base_key + "|in:" + std::to_string(i);
                bool report = false;
                {
                    std::lock_guard<std::mutex> mk(mt_mu);
                    mt_total++;
                    mt_count[mkey]++;
                    report = (mt_total % 500) == 0; // 每 500 次 miss 汇报一次 top（含首次累积）
                }
                if (report) {
                    std::lock_guard<std::mutex> mk(mt_mu);
                    fprintf(stderr, "[BW-MISS-TOP] total=%llu\n", (unsigned long long)mt_total);
                    // 降序 top 8
                    std::vector<std::pair<uint64_t, std::string>> v;
                    v.reserve(mt_count.size());
                    for (auto& kv : mt_count) v.emplace_back(kv.second, kv.first);
                    std::sort(v.begin(), v.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
                    for (size_t k = 0; k < v.size() && k < 8; ++k)
                        fprintf(stderr, "  %llu  hasKey=%d  %s\n",
                                (unsigned long long)v[k].first,
                                (int)C3KernelRegistry::getInstance().hasBackwardKey(v[k].second),
                                v[k].second.c_str());
                }
            }
            bemark(3); // phase1 段结束（此调用命中 miss）→ 进入 miss 段
            compileBackwardAsyncForInput(node, grad, i);
            // [P0.6B 2026-08-30 苏璃珞 重做] miss 后等所有 in-flight async 编译完成
            //
            // 历史：之前 compileBackwardAsyncForInput 启动 std::thread + .detach()
            //       不等完成。miss 路径立即 return std::nullopt。**下次同 key 调用**
            //       时大概率前一次 async 还在编译（5-50ms）→ backward_entries_ 仍空
            //        → 重复 fallback。实测覆盖率 6.25%。
            //
            // 修复（最安全方案，不改任何函数体）：miss 后**同步等**所有 in-flight
            //       编译任务完成。**主线程阻塞** 5-50ms × N（in-flight 数），
            //       但**之后**同 key 必命中。
            //
            // 不改 compileBackwardAsyncForInput 函数体：避免触发 static const 初始化
            //       时序问题（之前 P0.6B inline 同步版本 hang 根因未知）。
            {
                // [PHASE1-MISS 2026-09-02] C3_PH1_MISS=1：量化非 ReLU 节点 miss 后
                //   compileBackwardAsyncForInput + waitForPendingCompiles 的同步等待开销。
                static const bool ph1_miss = [] {
                    const char* e = std::getenv("C3_PH1_MISS");
                    return e && std::string(e) == "1";
                }();
                if (ph1_miss) {
                    static std::atomic<uint64_t> s_compile_us{0}, s_wait_us{0}, s_calls{0};
                    auto t0 = std::chrono::steady_clock::now();
                    compileBackwardAsyncForInput(node, grad, i);
                    auto t1 = std::chrono::steady_clock::now();
                    getInstance().waitForPendingCompiles();
                    auto t2 = std::chrono::steady_clock::now();
                    s_compile_us += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                    s_wait_us    += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                    s_calls++;
                    static size_t acc = 0;
                    if ((++acc) % 200 == 0)
                        fprintf(stderr, "[PHASE1-MISS] compile=%.2fms/%llu wait=%.2fms/%llu calls=%llu\n",
                                s_compile_us.load()*1e-3, (unsigned long long)s_calls.load(),
                                s_wait_us.load()*1e-3, (unsigned long long)s_calls.load(),
                                (unsigned long long)s_calls.load());
                } else {
                    compileBackwardAsyncForInput(node, grad, i);
                    getInstance().waitForPendingCompiles();
                }
                bemark(4); // miss 段结束（compile+wait 完成）→ 返回前
            }
            // [FIX 2026-09-03 苏璃珞] 编译同步完成后重试 execute，使首次调用即同步用 C3 结果，
            // 消除 miss→eager 回退与后台编译竞态导致的梯度未完整落盘（见 C3-BUG-20260903-01）。
            {
                auto retry = C3KernelRegistry::getInstance().tryExecuteBackward(
                    base_key + "|in:" + std::to_string(i), grad, forward_inputs);
                if (retry.has_value() && !retry->empty()) {
                    out.push_back(std::move(retry->at(0)));
                    continue; // 该输入已用 C3 内核同步算完，继续处理下一输入
                }
            }
            return std::nullopt; // 重试仍 miss → 整体 eager 回退
        }
        out.push_back(std::move(result->at(0)));
    }

    bemark(3); // phase1 段结束（未命中 miss，全部输入都取到 kernel）→ 进入 wrap 段
    bemark(5); // wrap 段结束（累积本身很短，仅作调用计数）

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
        if (!taskStarted()) return;
        // 【修复】不能用 std::async → future 析构会阻塞 → 变相同步
        std::thread([this, type, i, grad_desc, input_descs, per_key]() {
            struct TaskGuard { C3BackwardCapture* self; ~TaskGuard() { self->taskFinished(); } } guard{this};
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

    if (!taskStarted()) return;
    // 【修复】不能用 std::async → future 析构会阻塞 → 变相同步
    std::thread([this, type, input_index, grad_desc, input_descs, per_key]() {
        struct TaskGuard { C3BackwardCapture* self; ~TaskGuard() { self->taskFinished(); } } guard{this};
        // [BW-DIAG 2026-09-02] C3_BW_DIAG=1：定位「支持但装不上」的 per-key 编译失败原因
        static const bool bw_diag = [] {
            const char* e = std::getenv("C3_BW_DIAG");
            return e && std::string(e) == "1";
        }();
        static std::mutex diag_mu;
        static std::unordered_set<std::string> diag_seen;
        auto diag = [&](const char* reason) {
            if (!bw_diag) return;
            std::lock_guard<std::mutex> k(diag_mu);
            if (!diag_seen.insert(per_key).second) return;
            fprintf(stderr, "[BW-DIAG] key=%s reason=%s\n", per_key.c_str(), reason);
        };
        auto graph_opt = buildBackwardGraphForTypeAndIndex(type, input_index, grad_desc, input_descs);
        if (!graph_opt.has_value()) {
            diag("buildGraph=nullopt");
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_compiles_.erase(per_key);
            return;
        }
        auto& graph_pair = graph_opt.value();
        Graph& graph = graph_pair.first;
        const std::vector<size_t>& fwd_input_map = graph_pair.second;
        if (graph.nodeCount() <= graph.inputCount()) {
            diag("no-compute (nodeCount<=inputCount)");
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
            } else {
                diag("compile=kernel-null");
            }
        } catch (const std::exception& e) {
            diag(std::string("compile-threw: ").append(e.what()).c_str());
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
        // Softmax backward 暂保守回退 eager；多归约图仍未通过生命周期验证。
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
           // Sigmoid backward 暂回退 eager，待 live-range coloring 稳定后重新启用。

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
           // Softmax backward 暂回退 eager，避免多归约临时 buffer 生命周期风险。

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
    if (!taskStarted()) return;
    std::thread([this, seq, fused_key, reg_grad_shape, reg_input_shape]() {
        struct TaskGuard { C3BackwardCapture* self; ~TaskGuard() { self->taskFinished(); } } guard{this};
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
    // [PEL25 audit P0-3 2026-09-05] 多输出 backward fusion 主动禁用 — 见
    // reports/2026-09-05/code-review-c3-deep-audit-162100.md §3 P0-3
    //
    // 根因 (C3-BUG-20260905-01): ReLU→Sigmoid / ReLU→ReLU 等 element-wise 链
    // 在异步 kernel 安装后,intercepted 机制把错误的 upstream grad 回填到
    // 后续节点 → 训练数值错误。
    //
    // 当前策略 (correctness > hit rate):
    //   - 本函数始终返回 std::nullopt,所有反向融合都走单节点 backward C3 路径
    //   - 编译侧 (compileFusedBackwardAsync / recordBackwardNode / sequence_counts_)
    //     仍 populate fused_entries_,因为驱动它们的反向链统计是有用的 profiling 信号
    //   - 真根因 (DEBT-2) 在 tryExecuteFusedBackward 的 pending_intercepted_ 节点
    //     生命周期 + shape 校验路径,需重做 intercepted 调度才能恢复融合
    //
    // 重新启用条件 (PEL26+ TODO):
    //   1. 修 DEBT-2: chain_forward_inputs 构造 / installBackward shape 校验 /
    //      pending_intercepted_ 节点生命周期
    //   2. 加数值正确性回归测试:ReLU→Sigmoid / ReLU→ReLU 链 forward+backward,
    //      对比 C3 fused vs eager 梯度,差 < 1e-5
    //   3. 灰度启用 (C3_FUSED_BW=1 环境开关),先 1 epoch 验证 loss 与 eager 一致
    //
    // [PEL25 mitigation 2026-09-05] 加 C3_FUSED_BW 环境变量 kill switch:
    //   - C3_FUSED_BW=0 (默认): 返回 nullopt,行为完全不变 (本次仍默认 off)
    //   - C3_FUSED_BW=1: 记录 attempt 日志 + 仍返回 nullopt (因为原实现被删,PEL26+ 重新实装)
    //   - C3_FUSED_BW=2: 同 1 + 额外打印 intercepted entries (调试用)
    // 目的: 给 PEL26+ 修 DEBT-2 时一个明确的"我正在尝试"开关,无需重读代码
    static const int kFusedBwLevel = []() {
        const char* e = std::getenv("C3_FUSED_BW");
        if (!e) return 0;
        if (std::string(e) == "1") return 1;
        if (std::string(e) == "2") return 2;
        return 0;  // 任何其他值默认 off
    }();
    if (kFusedBwLevel >= 1) {
        // [DEBT-2 diagnostic] 记 attempt,表明有人在尝试 re-enable 但功能未实装
        static std::atomic<uint64_t> s_attempts{0};
        uint64_t n = s_attempts.fetch_add(1, std::memory_order_relaxed) + 1;
        if (kFusedBwLevel >= 2) {
            CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
                ErrorType::UNKNOWN,
                "[C3-DEBT-2] C3_FUSED_BW=" + std::to_string(kFusedBwLevel) +
                " attempt #" + std::to_string(n) +
                " — fused BW still disabled, set C3_FUSED_BW=0 to suppress");
        }
    }
    (void)node;
    (void)grad;
    (void)forward_inputs;
    return std::nullopt;
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
    // [MIMO-SEG 2026-09-02] C3_MIMO_SEG=1：稳态 HIT 命中路径内部五段耗时分布
    //   checks / keybuild / hitlookup / inbuild(输入向量拷贝) / exec / wrap(输出+插表)。
    //   定位 mimo guard ~600ms/ep 中「非内核」的编排开销到底出在哪一段。
    static const bool mimo_seg = [] {
        const char* e = std::getenv("C3_MIMO_SEG");
        return e && std::string(e) == "1";
    }();
    static struct { std::atomic<uint64_t> n[6], c[6]; } mseg;
    auto tprev = std::chrono::steady_clock::now();
    auto mark = [&tprev](int seg) {
        if (!mimo_seg) return;
        auto now = std::chrono::steady_clock::now();
        uint64_t d = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - tprev).count();
        mseg.n[seg] += d; mseg.c[seg]++;
        tprev = now;
        static thread_local size_t acc = 0;
        if ((++acc) % 300 == 0) {
            const char* nm[6] = {"checks","keybuild","hitlookup","inbuild","exec","wrap"};
            fprintf(stderr, "[MIMO-SEG]");
            for (int i = 0; i < 6; ++i)
                fprintf(stderr, " %s=%.2fms/%llu", nm[i], mseg.n[i]*1e-3, (unsigned long long)mseg.c[i]);
            fprintf(stderr, "\n");
        }
    };

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

    mark(0); // checks: typeid + 上游匹配 + 张量取址

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
    mark(1); // keybuild

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
        mark(2); // hitlookup: hasBackwardKey

        // 收集输入：[z, X, W]，不重复包含 grad，配合 fwd_input_map {0, 1, 2}
        std::vector<Tensor> inputs = {z, X, W};
        mark(3); // inbuild: 输入向量拷贝

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
            mark(4); // exec: kernel 实际执行

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
            mark(5); // wrap: 输出 vector + 插 pending 表
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

    if (!taskStarted()) return;
    std::thread([this, current_type, mimo_key, grad_desc, z_desc, x_desc, w_desc]() {
        struct TaskGuard { C3BackwardCapture* self; ~TaskGuard() { self->taskFinished(); } } guard{this};
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