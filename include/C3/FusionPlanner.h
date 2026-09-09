/**
 * @file FusionPlanner.h
 * @brief 通用图融合决策层 (Universal Graph Fusion Planner)
 * @details 把 C3「为每种结构手写一个融合 pass」收敛为数据驱动判据：
 *          在统一 Graph IR 上按 (a) lowering 能力、(b) shape/numel、(c) 数据依赖割
 *          自动产出可融合单元，取代逐结构的 if-else / 专有 MIMO 目录。
 * @date 2026-09-07 (scope: docs/C3_UNIVERSAL_FUSION_DESIGN.md)
 *
 * 设计原则（与项目红线一致）：
 *  - 纯函数、只读 Graph、不触发编译/执行；任何时刻可安全旁路运行。
 *  - 判据是「算子类别 + numel + 单消费者」，绝不按结构名特判（能力泛化）。
 *  - 保守方向：不确定就割开（多物化比错融合安全），保证正确性不回归。
 */

#ifndef CTORCH_C3_FUSION_PLANNER_H
#define CTORCH_C3_FUSION_PLANNER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Graph.h"

namespace ct {
namespace c3 {

/// 融合单元的 kernel 类别（lowering 能力，与「哪个算子能并」解耦）
enum class FusionUnitKind : uint8_t {
    ELEMENTWISE = 0,     ///< 纯逐元素单元：所有成员等 numel、identity-1D ABI 安全
    GEMM = 1,            ///< 仅一个 MatMul，无被吸收的 epilogue
    GEMM_EPILOGUE = 2,   ///< 一个 MatMul + 其单消费者逐元素尾链
    REGION_KERNEL = 3,   ///< 单内核多输出 region：连通区段内共享中间量只算一次
    LEAF = 4             ///< 结构性节点 / 图输入占位：自身一个 kernel，作物化边界
};

/// 融合策略（判据粒度，均数据驱动、非按结构名特判）
enum class FusionStrategy : uint8_t {
    Default = 0,   ///< 前向单 GEMM / 逐元素单元：多消费者中间量物化，双 GEMM 不并
    RegionKernel = 1 ///< 多输出 region：连通区段(共享中间量仅区段内复用)可并单内核；
                     ///<   Transpose 并入 GEMM；SumReduce/Softmax/CrossEntropy/Fused 仍为边界
};

/**
 * @brief RegionKernel 策略的跨分量合并度量（纯数据驱动）
 * @details 连通分量之外, 仅当分量间存在共享外部输入(同一 backward 调用派生的分支)才考虑合并:
 *          merged = has_shared_ext && (saved_reload_bytes + saved_launch_bytes)
 *                    > min_benefit_ratio * working_set_bytes。
 *          working_set 为 live 中间量(graph 输出除外, 因它们无论如何都要写回)。
 *          launch 项与系数终态由 autotune 机器指纹校准。
 */
struct RegionMergeMetric {
    size_t component_count = 0;        ///< 连通分量数
    uint64_t saved_reload_bytes = 0;   ///< Σ (usage(e)-1) * numel(e), 共享外部输入因并入单内核省的重读
    uint64_t saved_launch_bytes = 0;   ///< (k-1) * launch_unit_bytes, 仅在 has_shared_ext 时计入
    uint64_t working_set_bytes = 0;    ///< Σ 非 graph-output 的 regionable 节点 numel(live 中间量代理)
    bool merged = false;               ///< 是否把多分量并成单 region
};

/// RegionKernel 跨分量合并的代价门参数（原型默认保守; 系数终态由 autotune 指纹给出）
struct RegionFusionPolicy {
    double min_benefit_ratio = 0.25;       ///< 收益须至少覆盖工作集 25% 才跨分量合并
    uint64_t launch_unit_bytes = 400 * 1024; ///< 单次 launch 等价字节税(约 2µs @ 200GB/s), 终态由 autotune 校准
};

/// 单个融合单元
struct FusionUnit {
    FusionUnitKind kind = FusionUnitKind::LEAF;
    size_t unit_index = 0;            ///< 在 FusionPlan::units 中的下标
    std::vector<size_t> node_ids;     ///< 单元内节点 ID（升序，拓扑可执行）
    std::vector<size_t> external_input_ids; ///< 单元外部喂入的节点 ID（边界输入，去重升序）
    std::vector<size_t> output_ids;   ///< 单元对外输出节点 ID（图输出或多消费者物化点）
    size_t numel = 0;                 ///< 单元公共 numel（同单元经判据保证一致）

    bool isCompute() const { return kind != FusionUnitKind::LEAF; }
};

/// 一次规划的完整结果
struct FusionPlan {
    std::vector<FusionUnit> units;    ///< 所有单元（含 LEAF 单节点）
    std::vector<size_t> node_unit;    ///< node_id -> units 下标（size = nodeCount）
    size_t compute_unit_count = 0;    ///< 非 LEAF 单元数
    RegionMergeMetric region_metric;  ///< RegionKernel 策略的跨分量合并度量（Default 策略为零值）

    /// 取包含指定 node_id 的单元（node_id 非法则返回 nullptr）
    const FusionUnit* unitOf(size_t node_id) const {
        if (node_id >= node_unit.size()) return nullptr;
        size_t idx = node_unit[node_id];
        if (idx >= units.size()) return nullptr;
        return &units[idx];
    }
};

/**
 * @class FusionPlanner
 * @brief 融合决策：输入统一 Graph，输出一组可独立 kernel 化的融合单元。
 * @details 判据（全部由数据驱动，非结构特判）：
 *   1. lowering 能力：逐元素族(Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh/Gt/Exp/Log)
 *      可彼此共内核；MatMul 可吸收其单消费者逐元素尾链成 GEMM_EPILOGUE；
 *      SumReduce/Transpose/Softmax/CrossEntropy/Fused/Const 为结构性边界。
 *   2. shape/numel：纯逐元素单元要求成员 out numel 全等（identity-1D ABI 门，
 *      与 LinalgOneShot 收紧一致）；GEMM_EPILOGUE 的逐元素尾链须与 GEMM 输出等 numel。
 *   3. 数据依赖割：生产者多消费者时必须物化（不并入任一消费者）；
 *      GEMM 不并入 GEMM（不做共享 GEMM 合并——M3 探针已证负收益）。
 *  @note 保守：任何不确定处直接割开，保证正确性不回归。
 */
class FusionPlanner {
public:
    /// 默认前向策略（单 GEMM / 逐元素单元）
    static FusionPlan planUnits(const Graph& graph);

    /// 按指定策略规划融合单元
    /// @param strategy FusionStrategy::RegionKernel 时：多输出 region 判据
    ///                 （连通区段 + 共享中间量内联 + Transpose 并入 GEMM
    ///                 + 代价门跨分量合并, 度量见 plan.region_metric）。
    /// @param policy   RegionKernel 跨分量合并代价门参数（Default 策略忽略）
    static FusionPlan planUnits(const Graph& graph, FusionStrategy strategy,
                                const RegionFusionPolicy& policy = {});

    /// 供测试/诊断：节点 → 可共内核类别（不触发任何编译）
    static FusionUnitKind nodeKind(const Node& node);

private:
    FusionPlanner() = default;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_FUSION_PLANNER_H
