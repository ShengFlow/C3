/**
 * @file FusionPlanner.cpp
 * @brief 通用图融合决策层实现（数据驱动，非结构特判）
 * @date 2026-09-07 (scope: docs/C3_UNIVERSAL_FUSION_DESIGN.md)
 */

#include "C3/FusionPlanner.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace ct {
namespace c3 {

namespace {

size_t nodeNumel(const Node& n) {
    if (n.out_desc.numel > 0) return n.out_desc.numel;
    size_t v = 1;
    for (size_t d : n.out_desc.shape) v *= d;
    return v;
}

/// 该节点是否为「结构性 / 物化边界」类别（不进逐元素 / GEMM 尾链融合）
bool isStructural(const Node& n) {
    return std::visit([](auto&& op) -> bool {
        using T = std::decay_t<decltype(op)>;
        return std::is_same_v<T, ConstNode> ||      // 常量 / 图输入占位
               std::is_same_v<T, SumReduceNode> ||  // 降维，独立 kernel
               std::is_same_v<T, TransposeNode> ||  // 转置，独立 kernel
               std::is_same_v<T, SoftmaxNode> ||    // softmax，独立 kernel
               std::is_same_v<T, CrossEntropyNode> || // 交叉熵，独立 kernel
               std::is_same_v<T, FusedNode>;        // 已是融合单元（不透明）
    }, n.op);
}

} // namespace

FusionUnitKind FusionPlanner::nodeKind(const Node& node) {
    if (isStructural(node)) return FusionUnitKind::LEAF;
    return std::visit([](auto&& op) -> FusionUnitKind {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, MatMulNode>) {
            return FusionUnitKind::GEMM;
        }
        // 逐元素族：可彼此共内核（等 numel、identity-1D ABI）
        if constexpr (std::is_same_v<T, AddNode> || std::is_same_v<T, SubNode> ||
                      std::is_same_v<T, MulNode> || std::is_same_v<T, DivNode> ||
                      std::is_same_v<T, NegNode> || std::is_same_v<T, ReLUNode> ||
                      std::is_same_v<T, SigmoidNode> || std::is_same_v<T, TanhNode> ||
                      std::is_same_v<T, GtNode> || std::is_same_v<T, ExpNode> ||
                      std::is_same_v<T, LogNode>) {
            return FusionUnitKind::ELEMENTWISE;
        }
        return FusionUnitKind::LEAF;
    }, node.op);
}

FusionPlan FusionPlanner::planUnits(const Graph& graph) {
    const size_t n = graph.nodeCount();
    FusionPlan plan;
    plan.units.clear();
    plan.node_unit.assign(n, SIZE_MAX);

    const auto& nodes = graph.nodes();
    const auto& graph_inputs = graph.inputs();
    const auto& graph_outputs = graph.outputs();

    // 每个节点是否图输入（图输入 = 外部参数/占位，恒为边界）
    std::vector<bool> is_input(n, false);
    for (size_t id : graph_inputs) if (id < n) is_input[id] = true;

    std::vector<FusionUnitKind> cat(n, FusionUnitKind::LEAF);
    std::vector<bool> compute(n, false);
    for (size_t i = 0; i < n; ++i) {
        FusionUnitKind k = is_input[i] ? FusionUnitKind::LEAF
                                       : FusionPlanner::nodeKind(nodes[i]);
        cat[i] = k;
        compute[i] = (k != FusionUnitKind::LEAF);
    }

    // ---- union-find（仅合并 compute 节点） ----
    std::vector<size_t> parent(n);
    std::vector<size_t> rank(n, 0);
    std::vector<bool> hasGemm(n, false);       // 仅对根有意义
    std::vector<size_t> aggNumel(n, 0);        // 仅对根有意义（governing numel）
    for (size_t i = 0; i < n; ++i) {
        parent[i] = i;
        aggNumel[i] = nodeNumel(nodes[i]);
        hasGemm[i] = (cat[i] == FusionUnitKind::GEMM);
    }
    std::function<size_t(size_t)> find = [&](size_t x) -> size_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // 路径压缩
            x = parent[x];
        }
        return x;
    };

    // 尝试融合有向边 p -> c。规则见头文件；保守：任何不确定即割。
    auto tryMerge = [&](size_t p, size_t c) -> void {
        if (p == c) return;
        if (!compute[p] || !compute[c]) return;
        // 逐元素消费者才能并入上游；MatMul 可被逐元素尾链吸收
        if (cat[c] != FusionUnitKind::ELEMENTWISE) return;
        if (cat[p] != FusionUnitKind::ELEMENTWISE &&
            cat[p] != FusionUnitKind::GEMM) return;
        // 多消费者：必须物化，不并入任一消费者
        if (nodes[p].outputs.size() != 1) return;
        // 显式避免 GEMM 并入 GEMM / 双 GEMM 同单元（共享 GEMM 合并负收益，已证）
        if (cat[p] == FusionUnitKind::GEMM && cat[c] == FusionUnitKind::GEMM) return;

        size_t rp = find(p), rc = find(c);
        if (rp == rc) return;
        // 两个集合都已含 GEMM → 会变成双 GEMM 单元，禁止
        if (hasGemm[rp] && hasGemm[rc]) return;
        // numel 一致：纯逐元素要求成员同 numel；GEMM 单元要求尾链 == GEMM 输出 numel
        if (aggNumel[rp] != aggNumel[rc]) return;

        // 合并（按秩）
        if (rank[rp] < rank[rc]) std::swap(rp, rc);
        parent[rc] = rp;
        if (rank[rp] == rank[rc]) rank[rp]++;
        hasGemm[rp] = hasGemm[rp] || hasGemm[rc];
        // aggNumel 在允许分支时已保证相等
    };

    // 处理边：按生产者升序，保证前置子图先定型
    for (size_t p = 0; p < n; ++p) {
        if (!compute[p]) continue;
        for (size_t c : nodes[p].outputs) {
            if (c >= n) continue;
            tryMerge(p, c);
        }
    }

    // ---- 汇总集合 → 单元 ----
    // 先收集 compute 根 → 单元 kind（SetAgg 值初始化，避免未初始化字段）
    struct SetAgg { bool has_elem = false; bool has_gemm = false; size_t min_id = SIZE_MAX; };
    std::unordered_map<size_t, SetAgg> root_map;
    for (size_t i = 0; i < n; ++i) {
        if (!compute[i]) continue;
        size_t r = find(i);
        SetAgg& agg = root_map[r];
        agg.has_elem = agg.has_elem || (cat[i] == FusionUnitKind::ELEMENTWISE);
        agg.has_gemm = agg.has_gemm || (cat[i] == FusionUnitKind::GEMM);
        if (i < agg.min_id) agg.min_id = i;
    }
    // 决定 kind（按 min_id 排序保证确定性）
    std::vector<std::pair<size_t, size_t>> ordered; // (min_id, root)
    for (auto& [r, agg] : root_map) ordered.emplace_back(agg.min_id, r);
    std::sort(ordered.begin(), ordered.end());

    std::unordered_map<size_t, size_t> root_to_unit;
    plan.compute_unit_count = 0;
    for (auto& [min_id, r] : ordered) {
        const auto& agg = root_map[r];
        FusionUnitKind kind =
            agg.has_gemm ? (agg.has_elem ? FusionUnitKind::GEMM_EPILOGUE
                                         : FusionUnitKind::GEMM)
                         : FusionUnitKind::ELEMENTWISE;
        root_to_unit[r] = plan.units.size();
        FusionUnit u;
        u.kind = kind;
        u.unit_index = plan.units.size();
        u.numel = aggNumel[r]; // 判据已保证单元内 numel 一致
        // 收集成员
        for (size_t i = 0; i < n; ++i) {
            if (compute[i] && find(i) == r) u.node_ids.push_back(i);
        }
        plan.units.push_back(std::move(u));
        plan.compute_unit_count++;
    }

    // LEAF（结构性/图输入）各占一个单节点单元，附加在 compute 单元之后
    for (size_t i = 0; i < n; ++i) {
        if (compute[i]) continue;
        FusionUnit u;
        u.kind = FusionUnitKind::LEAF;
        u.unit_index = plan.units.size();
        u.node_ids.push_back(i);
        u.numel = nodeNumel(nodes[i]);
        root_to_unit[i] = plan.units.size();
        plan.units.push_back(std::move(u));
    }

    // 填充 node_unit + 边界信息
    for (size_t i = 0; i < n; ++i) {
        size_t r = compute[i] ? find(i) : i;
        size_t uidx = root_to_unit[r];
        plan.node_unit[i] = uidx;
    }

    auto isOutputNode = [&](size_t id) {
        return std::find(graph_outputs.begin(), graph_outputs.end(), id) != graph_outputs.end();
    };

    for (auto& u : plan.units) {
        std::unordered_set<size_t> ext_in;
        std::unordered_set<size_t> outs;
        std::vector<bool> member(n, false);
        for (size_t id : u.node_ids) member[id] = true;
        for (size_t id : u.node_ids) {
            const auto& nd = nodes[id];
            for (size_t in_id : nd.inputs) {
                if (in_id >= n || member[in_id]) continue; // 单元内 / 无效
                ext_in.insert(in_id);
            }
            if (isOutputNode(id)) { outs.insert(id); continue; }
            for (size_t out_id : nd.outputs) {
                if (out_id < n && !member[out_id]) { outs.insert(id); break; }
            }
        }
        u.external_input_ids.assign(ext_in.begin(), ext_in.end());
        std::sort(u.external_input_ids.begin(), u.external_input_ids.end());
        u.output_ids.assign(outs.begin(), outs.end());
        std::sort(u.output_ids.begin(), u.output_ids.end());
    }

    return plan;
}

} // namespace c3
} // namespace ct
