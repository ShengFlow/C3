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
 *   6. [AMX Micro-Kernel Phase 4] Apple Silicon AMX 寄存器级 Online Softmax & 融合 SDPA (AmxFlashAttentionEngine)
 *   7. [AMX Micro-Kernel Phase 4] Apple Silicon AMX 集群亲和性 MpGEMM 与异构 AMX+NEON 流水线 (AmxClusterMpGemmEngine)
 *   8. [AMX Micro-Kernel Phase 4] Apple Silicon AMX 片上 INT4 零访存解量化与 SwiGLU 融合超算子 (AmxDequantFusionEngine)
 *   9. [FlashAttention-3 FP8 Phase 4] 硬件异步双缓冲流水线与 E4M3 FP8 算子 (FlashAttention3Fp8Engine)
 */

#include "RamAdCurvatureEngine.h"
#include "RamAdDynamicRaggedEngine.h"
#include "NumaStaticMemoryPlanner.h"
#include "C3IntegratedCompiler.h"
#include "C3MNISTReplacementEngine.h"
#include "AmxFlashAttentionEngine.h"
#include "AmxClusterMpGemmEngine.h"
#include "AmxDequantFusionEngine.h"
#include "FlashAttention3Fp8Engine.h"

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
            "C3JIT::MIMO_BackwardFusion (Multi-Input Multi-Output In-Register Adjoint)",
            "AMX::FlashAttention (Apple Silicon AMX In-Register Online Softmax & Fused SDPA)",
            "AMX::ClusterMpGEMM (Apple Silicon AMX Cluster-Affinity MpGEMM & Heterogeneous Pipeline)",
            "AMX::DequantFusion (Apple Silicon AMX INT4 Dequantization & SwiGLU Fused Super-Kernel)",
            "FlashAttention3::FP8 (Hardware-Asynchronous Pipelining & E4M3 FP8 Micro-Kernel)"
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
        const typename RamAdCurvatureEngine<T>::CurvatureConfig& cfg,
        std::span<T> workspace = {})
    {
        RamAdCurvatureEngine<T>::execute_curvature_kfac(
            X, W, dY, V,
            out_Y, out_dW, out_A, out_S, out_FVP, cfg, workspace);
    }

    static void execute_amx_flash_attention(
        std::span<const float> Q,
        std::span<const float> K,
        std::span<const float> V,
        std::span<float> O,
        const AmxFlashAttentionEngine::FlashAttentionConfig& cfg)
    {
        AmxFlashAttentionEngine::execute_fused_sdpa(Q, K, V, O, cfg);
    }

    static void execute_amx_cluster_mpgemm(
        std::span<const float> X,
        std::span<const float> Wg,
        std::span<const float> Wu,
        std::span<const float> Wd,
        std::span<const float> R,
        std::span<float> Y,
        const AmxClusterMpGemmEngine::MpGemmConfig& cfg) noexcept
    {
        AmxClusterMpGemmEngine::execute_cluster_mpgemm(X, Wg, Wu, Wd, R, Y, cfg);
    }

    static void execute_amx_dequant_fused(
        std::span<const float> X,
        std::span<const uint8_t> Wg_packed,
        std::span<const float> scale_g,
        std::span<const float> zp_g,
        std::span<const uint8_t> Wu_packed,
        std::span<const float> scale_u,
        std::span<const float> zp_u,
        std::span<const float> Wd,
        std::span<const float> R,
        std::span<float> Y,
        const AmxDequantFusionEngine::DequantConfig& cfg)
    {
        AmxDequantFusionEngine::execute_amx_dequant_fused(
            X, Wg_packed, scale_g, zp_g, Wu_packed, scale_u, zp_u, Wd, R, Y, cfg);
    }

    static void execute_flash_attention_3_fp8(
        std::span<const float> Q,
        std::span<const float> K,
        std::span<const float> V,
        std::span<float> O,
        const FlashAttention3Config& cfg)
    {
        FlashAttention3Fp8Engine::execute_fused_sdpa_fp8(Q, K, V, O, cfg);
    }
};

} // namespace c3
} // namespace ct
