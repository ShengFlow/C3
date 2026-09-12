/**
 * @file C3OrchestratedKernel.h
 * @brief G3 真接管：把按 planner 切分出的多子内核编排成单个 CompiledKernel
 * @details 实现 CompiledKernel 抽象接口（execute/cacheKey/targetDevice/workspaceBytes），
 *          execute() 按拓扑序执行多个子内核，上游输出喂给下游输入，最后按整图输出顺序返回。
 *          这样 C3KernelRegistry::installBackward / tryExecuteBackward 的调用约定完全不变，
 *          运行时路径零改动——registry 只认 std::shared_ptr<CompiledKernel> 调 execute()。
 *
 *          语义对齐 partitionGraph（G3 集成点）：子图 external 输入分三类
 *          - 图输入占位：execute() 的 inputs[i] 按 graph_input_orig_ids 映射
 *          - 图内 Const：构造时物化成常量张量（const_tensors_），execute() 复用
 *          - 上游子图输出：按拓扑序先执行上游，从 tensorByOrig 取
 * @date 2026/9/10
 */

#ifndef CTORCH_C3_C3_ORCHESTRATED_KERNEL_H
#define CTORCH_C3_C3_ORCHESTRATED_KERNEL_H

#include "C3/C3Engine.h"
#include "C3/FusionPlanner.h"
#include "C3/Graph.h"
#include "Tensor.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace ct {
namespace c3 {

/**
 * @class OrchestratedKernel
 * @brief 编排多个子内核为一个 CompiledKernel（G3 真接管的执行载体）
 */
class OrchestratedKernel : public CompiledKernel {
public:
    struct SubKernel {
        std::shared_ptr<CompiledKernel> kernel;
        std::vector<size_t> input_orig_ids;   ///< 子图 input[i] 对应的原图节点 id
        std::vector<size_t> output_orig_ids;  ///< 子图 output[i] 对应的原图节点 id
    };

    OrchestratedKernel(std::string cache_key,
                       std::vector<size_t> graph_input_orig_ids,
                       std::vector<size_t> graph_output_orig_ids,
                       std::vector<SubKernel> sub_kernels,
                       std::vector<size_t> topo_order,
                       std::unordered_map<size_t, Tensor> const_tensors)
        : cache_key_(std::move(cache_key)),
          graph_input_orig_ids_(std::move(graph_input_orig_ids)),
          graph_output_orig_ids_(std::move(graph_output_orig_ids)),
          sub_kernels_(std::move(sub_kernels)),
          topo_order_(std::move(topo_order)),
          const_tensors_(std::move(const_tensors)) {}

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        // 1. 建立 原图节点 id -> Tensor 映射：预置 Const，再填图输入
        std::unordered_map<size_t, Tensor> tensorByOrig = const_tensors_;
        const size_t nin = std::min(graph_input_orig_ids_.size(), inputs.size());
        for (size_t i = 0; i < nin; ++i)
            tensorByOrig[graph_input_orig_ids_[i]] = inputs[i];

        // 2. 按拓扑序执行子内核，上游输出喂给下游输入
        for (size_t k : topo_order_) {
            if (k >= sub_kernels_.size()) continue;
            const SubKernel& sk = sub_kernels_[k];
            std::vector<Tensor> sub_inputs;
            sub_inputs.reserve(sk.input_orig_ids.size());
            for (size_t oid : sk.input_orig_ids) {
                auto it = tensorByOrig.find(oid);
                if (it == tensorByOrig.end())
                    throw std::runtime_error("OrchestratedKernel: 子图输入 orig=" +
                                             std::to_string(oid) + " 未物化(缺失上游依赖?)");
                sub_inputs.push_back(it->second);
            }
            std::vector<Tensor> outs = sk.kernel->execute(sub_inputs);
            const size_t no = std::min(sk.output_orig_ids.size(), outs.size());
            for (size_t i = 0; i < no; ++i)
                tensorByOrig[sk.output_orig_ids[i]] = outs[i];
        }

        // 3. 按整图输出顺序返回
        std::vector<Tensor> result;
        result.reserve(graph_output_orig_ids_.size());
        for (size_t oid : graph_output_orig_ids_) {
            auto it = tensorByOrig.find(oid);
            if (it == tensorByOrig.end())
                throw std::runtime_error("OrchestratedKernel: 整图输出 orig=" +
                                         std::to_string(oid) + " 未覆盖");
            result.push_back(it->second);
        }
        return result;
    }

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }

    [[nodiscard]] DeviceType targetDevice() const override {
        for (const auto& sk : sub_kernels_)
            if (sk.kernel) return sk.kernel->targetDevice();
        return DeviceType::kCPU;
    }

    [[nodiscard]] size_t workspaceBytes() const override {
        size_t total = 0;
        for (const auto& sk : sub_kernels_)
            if (sk.kernel) total += sk.kernel->workspaceBytes();
        return total;
    }

private:
    std::string cache_key_;
    std::vector<size_t> graph_input_orig_ids_;
    std::vector<size_t> graph_output_orig_ids_;
    std::vector<SubKernel> sub_kernels_;
    std::vector<size_t> topo_order_;
    std::unordered_map<size_t, Tensor> const_tensors_;
};

/**
 * @brief 由 partitionGraph 的切分结果构建 OrchestratedKernel（G3 真接管工厂）
 * @param graph       原图（fused_graph）
 * @param subs        partitionGraph 切出的子图
 * @param sub_kernels 每个子图编译好的内核（顺序与 subs 一致）
 * @return 编排内核；若子图间无依赖也可用（单子图则退化为直通，仍正确）
 * @details 做三件事：① 按 upstream_units 拓扑排序 ② 物化 Const 外部输入
 *          ③ 组装 OrchestratedKernel。不触发任何编译（编译由调用方完成）。
 */
inline std::shared_ptr<OrchestratedKernel> buildOrchestratedKernel(
    const Graph& graph,
    const std::vector<PartitionedSubGraph>& subs,
    const std::vector<std::shared_ptr<CompiledKernel>>& sub_kernels) {
    if (subs.size() != sub_kernels.size()) return nullptr;

    // ① 拓扑排序（按 upstream 依赖）
    std::vector<size_t> order;
    std::vector<size_t> indeg(subs.size(), 0);
    std::vector<std::vector<size_t>> adj(subs.size());
    for (size_t k = 0; k < subs.size(); ++k)
        for (size_t up : subs[k].upstream_units) { adj[up].push_back(k); indeg[k]++; }
    std::queue<size_t> q;
    for (size_t k = 0; k < subs.size(); ++k) if (indeg[k] == 0) q.push(k);
    while (!q.empty()) {
        const size_t k = q.front(); q.pop();
        order.push_back(k);
        for (size_t nxt : adj[k]) if (--indeg[nxt] == 0) q.push(nxt);
    }
    if (order.size() != subs.size()) return nullptr;  // 依赖环(不应发生)

    // ② 物化 Const 外部输入（图内常量），避免 execute() 每次重建
    std::unordered_map<size_t, Tensor> const_tensors;
    const auto& orig_inputs = graph.inputs();
    const std::unordered_set<size_t> inputSet(orig_inputs.begin(), orig_inputs.end());
    for (const auto& s : subs) {
        for (size_t oid : s.input_orig_ids) {
            if (oid >= graph.nodeCount()) continue;
            if (inputSet.count(oid)) continue;              // 图输入占位, 运行时喂
            const NodeVariant& nv = graph.node(oid).op;
            if (!std::holds_alternative<ConstNode>(nv)) continue;  // 上游子图输出
            const double val = std::get<ConstNode>(nv).value;
            const auto& d = graph.node(oid).out_desc;
            Tensor t(ShapeTag{}, d.shape, DType::kFloat, DeviceType::kCPU);
            float* p = t.data_write<float>();
            for (size_t i = 0; i < t.numel(); ++i) p[i] = static_cast<float>(val);
            const_tensors.emplace(oid, std::move(t));
        }
    }

    // ③ 组装
    std::vector<OrchestratedKernel::SubKernel> sks;
    sks.reserve(subs.size());
    for (size_t k = 0; k < subs.size(); ++k) {
        OrchestratedKernel::SubKernel sk;
        sk.kernel = sub_kernels[k];
        sk.input_orig_ids = subs[k].input_orig_ids;
        sk.output_orig_ids = subs[k].output_orig_ids;
        sks.push_back(std::move(sk));
    }

    std::vector<size_t> gin = graph.inputs();
    std::vector<size_t> gout = graph.outputs();
    return std::make_shared<OrchestratedKernel>(
        // [Fix §4.97] cacheKey 加入子图结构摘要(原仅 "orchestrated|N" 无语义,
        // 未来 JITCache 键控会撞键): nodeCount 序列
        [&]() -> std::string {
            std::string ck = "orchestrated|" + std::to_string(subs.size());
            for (const auto& s : subs) ck += "|" + std::to_string(s.graph.nodeCount());
            return ck;
        }(), std::move(gin), std::move(gout),
        std::move(sks), std::move(order), std::move(const_tensors));
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3_ORCHESTRATED_KERNEL_H
