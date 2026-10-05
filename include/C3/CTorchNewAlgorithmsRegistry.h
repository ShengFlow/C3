#pragma once

/**
 * @file CTorchNewAlgorithmsRegistry.h
 * @brief CTorch C3 JIT: 新一代前沿算法全量集成注册表与运行时引擎 (Unified Algorithm Registry)
 * @details 将本阶段攻关的核心前沿算法全量挂载并无缝接入 CTorch 核心执行体系:
 *   1. [RAM-AD Chapter 15 / Theorem 19] 高阶曲率伴随与块对角 K-FAC 寄存器级单趟求导引擎 (RamAdCurvatureEngine)
 *   2. [RAM-AD Chapter 16 / Theorem 20] 变长 Ragged 序列与动态 Shape 符号内存规划之片上伴随求解引擎 (RamAdDynamicRaggedEngine)
 *   3. [Memory Planner Chapter 17 / Theorem 17] 非齐次多 NUMA 节点与 CXL 互联感知静态内存规划调度器 (NumaStaticMemoryPlanner)
 *   4. [Memory Planner Chapter 16 / Theorem 16] 计算图拓扑重排协同区间图着色静态复用调度器 (TRO-SMP)
 *   5. 端到端深度赋能 CTorch MNIST 训练网络与 JIT 执行管线 (TransparentMNISTNet / C3IntegratedCompiler)
 */

#include "RamAdCurvatureEngine.h"
#include "RamAdDynamicRaggedEngine.h"
#include "NumaStaticMemoryPlanner.h"
#include "C3IntegratedCompiler.h"
#include "C3MNISTReplacementEngine.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <span>
#include <cassert>

namespace ct {
namespace c3 {

class CTorchNewAlgorithmsHub {
public:
    [[nodiscard]] static std::vector<std::string> get_registered_algorithms() {
        return {
            "RAM-AD::CurvatureKFAC (Theorem 19: In-Register Curvature & FVP without Sample Tape)",
            "RAM-AD::DynamicRagged (Theorem 20: Symbolic Tiling & In-Register Masked Adjoint for Variable-Length)",
            "MemoryPlanner::NUMA-SMP (Theorem 17: NUMA Domain Partitioning & Cross-Socket Minimization)",
            "MemoryPlanner::TRO-SMP (Theorem 16: Topological Reordering & Liveness-Guided Scheduling)",
            "MemoryPlanner::IA-IGC (Theorem 8: Inplace-Aware Interval Graph Coloring with Super-Intervals)",
            "C3JIT::LockFreeDispatchTable (Atomic RCU Fast Pointer Resolution)",
            "C3JIT::MIMO_BackwardFusion (Multi-Input Multi-Output In-Register Adjoint)"
        };
    }

    static NumaStaticMemoryPlanner::PlanResult plan_numa_workload(
        std::span<NumaTensorDescriptor> tensors,
        size_t num_nodes = 2)
    {
        return NumaStaticMemoryPlanner::plan(tensors, num_nodes);
    }

    template <typename T = double>
    static void execute_ragged_sequence(
        std::span<const T> X_packed,
        std::span<const T> W1,
        std::span<const T> W2,
        std::span<const T> dY_packed,
        std::span<const size_t> offsets,
        std::span<T> out_Y,
        std::span<T> out_dX,
        std::span<T> out_dW1,
        std::span<T> out_dW2,
        const typename RamAdDynamicRaggedEngine<T>::RaggedConfig& cfg)
    {
        RamAdDynamicRaggedEngine<T>::execute_ragged_fused_ad(
            X_packed, W1, W2, dY_packed, offsets,
            out_Y, out_dX, out_dW1, out_dW2, cfg);
    }

    template <typename T = double>
    static void execute_kfac_curvature(
        std::span<const T> X,
        std::span<const T> W,
        std::span<const T> dY,
        std::span<const T> V,
        std::span<T> out_Y,
        std::span<T> out_dW,
        std::span<T> out_A,
        std::span<T> out_S,
        std::span<T> out_FVP,
        const typename RamAdCurvatureEngine<T>::CurvatureConfig& cfg)
    {
        RamAdCurvatureEngine<T>::execute_curvature_kfac(
            X, W, dY, V,
            out_Y, out_dW, out_A, out_S, out_FVP, cfg);
    }
};

} // namespace c3
} // namespace ct
