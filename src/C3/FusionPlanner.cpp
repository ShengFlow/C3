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

#include "C3/MachineFingerprint.h"
#include "C3/C3Config.h"   // forceRegionMergeEnabled()

namespace ct {
namespace c3 {

namespace {

size_t nodeNumel(const Node& n) {
    if (n.out_desc.numel > 0) return n.out_desc.numel;
    size_t v = 1;
    for (size_t d : n.out_desc.shape) v *= d;
    return v;
}

/// 结构性 / 物化边界（默认策略）：不并入逐元素 / GEMM 尾链融合
bool isStructuralDefault(const Node& n) {
    return std::visit([](auto&& op) -> bool {
        using T = std::decay_t<decltype(op)>;
        return std::is_same_v<T, ConstNode> ||      // 常量 / 图输入占位
               std::is_same_v<T, SumReduceNode> ||  // 降维
               std::is_same_v<T, TransposeNode> ||  // 默认：转置独立 kernel
               std::is_same_v<T, SoftmaxNode> ||
               std::is_same_v<T, CrossEntropyNode> ||
               std::is_same_v<T, FusedNode>;        // 已融合单元（不透明）
    }, n.op);
}

/// region 策略硬边界：连 region 也切开的 op（保留各自独立 kernel）
bool isRegionSeparator(const Node& n) {
    return std::visit([](auto&& op) -> bool {
        using T = std::decay_t<decltype(op)>;
        return std::is_same_v<T, ConstNode> ||
               std::is_same_v<T, SumReduceNode> ||
               std::is_same_v<T, SoftmaxNode> ||
               std::is_same_v<T, CrossEntropyNode> ||
               std::is_same_v<T, FusedNode>;
    }, n.op);
}

/// 由已构造好的单元集生成最终 FusionPlan（node_unit + 边界信息 + compute 计数）
FusionPlan assemble(const Graph& graph, std::vector<FusionUnit> units) {
    const size_t n = graph.nodeCount();
    FusionPlan plan;
    plan.units = std::move(units);
    plan.node_unit.assign(n, SIZE_MAX);
    plan.compute_unit_count = 0;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        FusionUnit& unit = plan.units[u];
        unit.unit_index = u;
        if (unit.isCompute()) plan.compute_unit_count++;
        for (size_t id : unit.node_ids)
            if (id < n) plan.node_unit[id] = u;
    }
    const auto& nodes = graph.nodes();
    const auto& graph_outputs = graph.outputs();
    auto isOutputNode = [&](size_t id) {
        return std::find(graph_outputs.begin(), graph_outputs.end(), id) != graph_outputs.end();
    };
    for (auto& u : plan.units) {
        std::unordered_set<size_t> ext_in;
        std::unordered_set<size_t> outs;
        std::vector<bool> member(n, false);
        for (size_t id : u.node_ids) if (id < n) member[id] = true;
        for (size_t id : u.node_ids) {
            const auto& nd = nodes[id];
            for (size_t in_id : nd.inputs) {
                if (in_id >= n || member[in_id]) continue;
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

// ======================= Default 策略（前向单 GEMM / 逐元素单元） =======================

FusionPlan planDefault(const Graph& graph) {
    const size_t n = graph.nodeCount();
    const auto& nodes = graph.nodes();
    const auto& graph_inputs = graph.inputs();

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

    std::vector<size_t> parent(n), rank(n, 0);
    std::vector<bool> hasGemm(n, false);
    std::vector<size_t> aggNumel(n, 0);
    for (size_t i = 0; i < n; ++i) {
        parent[i] = i;
        aggNumel[i] = nodeNumel(nodes[i]);
        hasGemm[i] = (cat[i] == FusionUnitKind::GEMM);
    }
    std::function<size_t(size_t)> find = [&](size_t x) -> size_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

    auto tryMerge = [&](size_t p, size_t c) -> void {
        if (p == c || !compute[p] || !compute[c]) return;
        if (cat[c] != FusionUnitKind::ELEMENTWISE) return;
        if (cat[p] != FusionUnitKind::ELEMENTWISE && cat[p] != FusionUnitKind::GEMM) return;
        if (nodes[p].outputs.size() != 1) return;                 // 多消费者物化
        if (cat[p] == FusionUnitKind::GEMM && cat[c] == FusionUnitKind::GEMM) return;
        size_t rp = find(p), rc = find(c);
        if (rp == rc) return;
        if (hasGemm[rp] && hasGemm[rc]) return;                   // 不合成双 GEMM
        if (aggNumel[rp] != aggNumel[rc]) return;                 // numel 一致性
        if (rank[rp] < rank[rc]) std::swap(rp, rc);
        parent[rc] = rp;
        if (rank[rp] == rank[rc]) rank[rp]++;
        hasGemm[rp] = hasGemm[rp] || hasGemm[rc];
    };

    for (size_t p = 0; p < n; ++p) {
        if (!compute[p]) continue;
        for (size_t c : nodes[p].outputs) if (c < n) tryMerge(p, c);
    }

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
    std::vector<std::pair<size_t, size_t>> ordered;
    for (auto& [r, agg] : root_map) ordered.emplace_back(agg.min_id, r);
    std::sort(ordered.begin(), ordered.end());

    std::vector<FusionUnit> units;
    for (auto& [min_id, r] : ordered) {
        const SetAgg& agg = root_map[r];
        FusionUnitKind kind = agg.has_gemm
            ? (agg.has_elem ? FusionUnitKind::GEMM_EPILOGUE : FusionUnitKind::GEMM)
            : FusionUnitKind::ELEMENTWISE;
        FusionUnit u;
        u.kind = kind;
        u.numel = aggNumel[r];
        for (size_t i = 0; i < n; ++i)
            if (compute[i] && find(i) == r) u.node_ids.push_back(i);
        units.push_back(std::move(u));
    }
    for (size_t i = 0; i < n; ++i) {
        if (compute[i]) continue;
        FusionUnit u;
        u.kind = FusionUnitKind::LEAF;
        u.numel = nodeNumel(nodes[i]);
        u.node_ids.push_back(i);
        units.push_back(std::move(u));
    }
    return assemble(graph, std::move(units));
}

// ======================= RegionKernel 策略（单内核多输出 region） =======================

FusionPlan planRegionKernel(const Graph& graph, const RegionFusionPolicy& policy) {
    const size_t n = graph.nodeCount();
    const auto& nodes = graph.nodes();
    const auto& graph_inputs = graph.inputs();

    std::vector<bool> is_input(n, false);
    for (size_t id : graph_inputs) if (id < n) is_input[id] = true;

    // 可 region 化：非输入 且 非硬边界 op。含 Transpose(region 内可折叠进 GEMM)、
    // 逐元素族、MatMul。SumReduce/Softmax/CrossEntropy/Fused/Const 为独立边界。
    std::vector<bool> regionable(n, false);
    for (size_t i = 0; i < n; ++i) {
        if (is_input[i]) continue;
        if (isRegionSeparator(nodes[i])) continue;
        regionable[i] = true;
    }

    // union-find：连通 regionable 分量并成同一 region
    //（共享中间量不物化、允许多 GEMM——单内核顺序执行 + 多输出）
    std::vector<size_t> parent(n), rank(n, 0);
    for (size_t i = 0; i < n; ++i) parent[i] = i;
    std::function<size_t(size_t)> find = [&](size_t x) -> size_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](size_t a, size_t b) {
        size_t ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rank[ra] < rank[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
    };
    for (size_t p = 0; p < n; ++p) {
        if (!regionable[p]) continue;
        for (size_t c : nodes[p].outputs) {
            if (c >= n || !regionable[c]) continue;
            unite(p, c);
        }
    }

    std::unordered_map<size_t, size_t> root_min; // root -> min node id
    for (size_t i = 0; i < n; ++i) {
        if (!regionable[i]) continue;
        size_t r = find(i);
        auto it = root_min.find(r);
        if (it == root_min.end() || i < it->second) root_min[r] = i;
    }
    std::vector<std::pair<size_t, size_t>> ordered; // (min_id, root)
    for (auto& [r, m] : root_min) ordered.emplace_back(m, r);
    std::sort(ordered.begin(), ordered.end());

    // ---- 跨分量合并代价门（数据驱动, 非按结构名） ----
    // 每个连通分量一组; 统计"外部输入被多少个分量消费"→ 并入单内核省的重读字节。
    std::vector<std::vector<size_t>> comp_members(ordered.size());
    for (size_t ci = 0; ci < ordered.size(); ++ci) {
        size_t r = ordered[ci].second;
        for (size_t i = 0; i < n; ++i)
            if (regionable[i] && find(i) == r) comp_members[ci].push_back(i);
    }
    std::unordered_map<size_t, size_t> ext_usage; // 外部输入节点 id -> 消费它的分量数
    std::vector<std::vector<size_t>> comp_ext(ordered.size());
    for (size_t ci = 0; ci < comp_members.size(); ++ci) {
        std::vector<bool> member(n, false);
        for (size_t id : comp_members[ci]) member[id] = true;
        for (size_t id : comp_members[ci]) {
            for (size_t in_id : nodes[id].inputs) {
                if (in_id >= n || member[in_id]) continue;
                // 每个分量对同一外部输入只计一次
                if (std::find(comp_ext[ci].begin(), comp_ext[ci].end(), in_id) == comp_ext[ci].end()) {
                    comp_ext[ci].push_back(in_id);
                    ext_usage[in_id] += 1;
                }
            }
        }
    }
    RegionMergeMetric metric;
    metric.component_count = comp_members.size();
    bool has_shared_ext = false;
    for (auto& [e, cnt] : ext_usage) {
        if (cnt <= 1) continue;
        has_shared_ext = true;
        metric.saved_reload_bytes += (uint64_t)(cnt - 1) * (uint64_t)nodeNumel(nodes[e]);
    }
    // live 峰值工作集: 任意时刻同时存活(已产未死)中间量的最大 numel。
    // graph 输出除外(无论如何写回), 只算真中间量; 比"求和"更准(求和会高估)。
    // 拓扑序用节点 id(递增, Graph 保证输入 id < 自身); 中间量 m live 于 [m, last_use(m)]。
    const auto& graph_outputs = graph.outputs();
    std::vector<bool> is_output(n, false);
    for (size_t o : graph_outputs) if (o < n) is_output[o] = true;

    struct Ev { size_t pos; int64_t delta; bool add; }; // add 先于 remove(同 pos)
    std::vector<Ev> events;
    for (size_t i = 0; i < n; ++i) {
        if (!regionable[i] || is_output[i]) continue;
        if (nodes[i].outputs.empty()) continue;      // 死中间量(不产生 live)
        size_t last_use = nodes[i].outputs[0];
        for (size_t c : nodes[i].outputs) if (c > last_use) last_use = c;
        events.push_back({i, (int64_t)nodeNumel(nodes[i]), true});
        events.push_back({last_use, -(int64_t)nodeNumel(nodes[i]), false});
    }
    std::sort(events.begin(), events.end(),
              [](const Ev& a, const Ev& b) {
                  return a.pos != b.pos ? a.pos < b.pos : (a.add && !b.add);
              });
    uint64_t peak = 0, running = 0;
    for (const Ev& e : events) {
        if (e.add) running += (uint64_t)e.delta;
        else       running -= (uint64_t)(-e.delta);
        if (running > peak) peak = running;
    }
    metric.working_set_bytes = peak;
    // regionable 节点总数(Allow 策略的规模保护依据)
    {
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) if (regionable[i]) cnt++;
        metric.region_node_count = cnt;
    }
    // launch 省税仅在同一次 backward 调用的分支间(共享外部输入)才计入
    if (has_shared_ext && metric.component_count > 1) {
        metric.saved_launch_bytes =
            (uint64_t)(metric.component_count - 1) * policy.launch_unit_bytes;
    }
    // 结构可并前提: 多分量 + 存在共享外部输入(同一次调用的分支间才有省重读/省 launch 语义)
    const bool structurally_mergeable = (metric.component_count > 1) && has_shared_ext;
    if (policy.force_merge) {
        // [强制合并] 跳过一切判定, 只保留结构前提。
        // 用途: 解耦"划分是否正确"(结构等价性, G1 一致率) 与"划分是否划算"(收益模型)。
        // 注意: saved_* / working_set / region_node_count 仍被填充, 供调用方观测。
        metric.merged = structurally_mergeable;
    } else if (policy.merge_strategy == RegionMergeStrategy::Allow) {
        // [ADR-0002 方案 C] 跨分量默认合并; 判别力下沉到"规模保护上限"。
        // 依据: 实测跨分量合并收益仅 0.1% 量级, 相对收益门槛在该层判别力价值低。
        metric.merged = structurally_mergeable &&
                        (metric.region_node_count <= policy.max_region_nodes);
    } else {
        // [Strict 现行策略] 相对收益门槛, 行为与 ADR-0002 之前完全一致。
        metric.merged = structurally_mergeable &&
                        ((metric.saved_reload_bytes + metric.saved_launch_bytes) >
                         (uint64_t)((double)metric.working_set_bytes * policy.min_benefit_ratio));
    }

    // ---- 组装单元 ----
    std::vector<FusionUnit> units;
    if (metric.merged) {
        FusionUnit u;
        u.kind = FusionUnitKind::REGION_KERNEL;
        u.numel = 0;
        for (size_t i = 0; i < n; ++i)
            if (regionable[i]) u.node_ids.push_back(i);
        units.push_back(std::move(u));
    } else {
        for (size_t ci = 0; ci < comp_members.size(); ++ci) {
            FusionUnit u;
            u.kind = FusionUnitKind::REGION_KERNEL;
            u.numel = 0;
            u.node_ids = comp_members[ci]; // 已按 id 升序(遍历升序收集)
            units.push_back(std::move(u));
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (regionable[i]) continue;
        FusionUnit u;
        u.kind = FusionUnitKind::LEAF;
        u.numel = nodeNumel(nodes[i]);
        u.node_ids.push_back(i);
        units.push_back(std::move(u));
    }
    FusionPlan plan = assemble(graph, std::move(units));
    plan.region_metric = metric;
    return plan;
}

} // namespace

FusionUnitKind FusionPlanner::nodeKind(const Node& node) {
    if (isStructuralDefault(node)) return FusionUnitKind::LEAF;
    return std::visit([](auto&& op) -> FusionUnitKind {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, MatMulNode>) {
            return FusionUnitKind::GEMM;
        }
        if constexpr (std::is_same_v<T, AddNode> || std::is_same_v<T, SubNode> ||
                      std::is_same_v<T, MulNode> || std::is_same_v<T, DivNode> ||
                      std::is_same_v<T, NegNode> || std::is_same_v<T, ReLUNode> ||
                      std::is_same_v<T, SigmoidNode> || std::is_same_v<T, TanhNode> ||
                      std::is_same_v<T, SiLUNode> ||
                      std::is_same_v<T, GtNode> || std::is_same_v<T, ExpNode> ||
                      std::is_same_v<T, LogNode>) {
            return FusionUnitKind::ELEMENTWISE;
        }
        return FusionUnitKind::LEAF;
    }, node.op);
}

FusionPlan FusionPlanner::planUnits(const Graph& graph) {
    return planUnits(graph, FusionStrategy::Default);
}

FusionPlan FusionPlanner::planUnits(const Graph& graph, FusionStrategy strategy,
                                    const RegionFusionPolicy& policy) {
    if (strategy == FusionStrategy::RegionKernel) {
        return planRegionKernel(graph, policy);
    }
    return planDefault(graph);
}

std::vector<PartitionedSubGraph> partitionGraph(const Graph& graph, const FusionPlan& plan) {
    std::vector<PartitionedSubGraph> subs;
    const size_t n = graph.nodeCount();

    // 原图节点 -> 所属 compute unit 的 plan.units 下标（用于子图间依赖检测）
    std::vector<size_t> node_unit_of(n, SIZE_MAX);
    for (size_t ui = 0; ui < plan.units.size(); ++ui) {
        if (!plan.units[ui].isCompute()) continue;
        for (size_t id : plan.units[ui].node_ids)
            if (id < n) node_unit_of[id] = ui;
    }

    for (size_t ui = 0; ui < plan.units.size(); ++ui) {
        const FusionUnit& u = plan.units[ui];
        if (!u.isCompute()) continue;   // LEAF(图输入/结构边界)只作子图外部输入

        PartitionedSubGraph sub;
        sub.unit_index = ui;
        sub.unit_node_ids = u.node_ids;

        const std::unordered_set<size_t> member(u.node_ids.begin(), u.node_ids.end());
        std::unordered_map<size_t, size_t> ext_to_input;   // 原图 id -> 子图 input id

        // node_ids 已按原图 id 升序（= 拓扑序），故 unit 内输入必已先添加
        for (size_t nid : u.node_ids) {
            const Node& nd = graph.node(nid);
            std::vector<size_t> sub_inputs;
            sub_inputs.reserve(nd.inputs.size());
            for (size_t in : nd.inputs) {
                if (member.count(in)) {
                    sub_inputs.push_back(sub.orig_to_sub.at(in));
                    continue;
                }
                auto it = ext_to_input.find(in);
                if (it == ext_to_input.end()) {
                    const TensorDesc d = graph.validNodeId(in) ? graph.node(in).out_desc
                                                               : TensorDesc{};
                    size_t sid = sub.graph.addInput(d);
                    ext_to_input.emplace(in, sid);
                    sub.input_orig_ids.push_back(in);
                    it = ext_to_input.find(in);
                }
                sub_inputs.push_back(it->second);
            }
            sub.orig_to_sub.emplace(nid, sub.graph.addNode(nd.op, sub_inputs, nd.out_desc));
        }

        for (size_t oid : u.output_ids) {
            auto it = sub.orig_to_sub.find(oid);
            if (it == sub.orig_to_sub.end()) continue;
            sub.graph.markOutput(it->second);
            sub.output_orig_ids.push_back(oid);
        }

        // 依赖检测：外部输入若来自另一个 compute unit 的输出，记为其 upstream
        // （units 按 min_id 排序 → 拓扑序，故上游子图通常已在 subs 中）
        for (size_t in : sub.input_orig_ids) {
            if (in >= n) continue;
            const size_t src_unit = node_unit_of[in];
            if (src_unit == SIZE_MAX) continue;   // 来自图输入/LEAF 边界
            for (size_t k = 0; k < subs.size(); ++k) {
                if (subs[k].unit_index == src_unit) {
                    sub.upstream_units.push_back(k);
                    break;
                }
            }
        }
        std::sort(sub.upstream_units.begin(), sub.upstream_units.end());
        sub.upstream_units.erase(
            std::unique(sub.upstream_units.begin(), sub.upstream_units.end()),
            sub.upstream_units.end());

        subs.push_back(std::move(sub));
    }
    return subs;
}

RegionFusionPolicy RegionFusionPolicy::fromMachineDefaults() {
    RegionFusionPolicy p;
    p.launch_unit_bytes = MachineFingerprint::instance().launchUnitBytes();
    // [强制合并] C3_FORCE_REGION_MERGE=1 时跳过收益门槛(代价判定后补), 默认关=现有行为
    p.force_merge = forceRegionMergeEnabled();
    // [ADR-0002 方案 C] C3_REGION_MERGE_ALLOW=1 时跨分量默认合并(仅受规模保护约束)
    p.merge_strategy = regionMergeAllowEnabled() ? RegionMergeStrategy::Allow
                                                 : RegionMergeStrategy::Strict;
    return p;
}

} // namespace c3
} // namespace ct
