/**
 * @file C3Config.h
 * @generation SHARED 跨代开关配置（所有 C3_* env var 集中于此）
 * @brief C3 子功能统一开关体系
 * @details 集中管理 C3 各子功能的开启/关闭，替代散落的 getenv 与硬编码宏。
 *          每个子功能支持两级控制：
 *          1. 编译期硬开关：由 CMake option 生成对应宏（如 CT_C3_DISABLE_SINGLE_KERNEL 生成
 *             C3_DISABLE_SINGLE_KERNEL_PP），编译期关闭后无法在运行时重新开启。
 *          2. 运行时软开关：通过环境变量（如 C3_DISABLE_SINGLE_KERNEL=1）在运行时关闭。
 *
 *          查询接口采用"编译期宏 OR 运行时 env"短路：只要任一层级关闭即视为关闭。
 *          所有查询结果缓存为 static，避免热路径重复 getenv / 分支开销。
 *
 * 子功能清单（对应宏 / env / 查询接口）：
 *  - 单 kernel hotpath 注入   : CT_C3_DISABLE_SINGLE_KERNEL / C3_DISABLE_SINGLE_KERNEL / singleKernelInjectionEnabled()
 *  - 区域融合 (Region Fusion) : CT_C3_DISABLE_REGION_FUSION / C3_DISABLE_REGION_FUSION / regionFusionEnabled()
 *  - 后向融合 (Backward)      : CT_C3_DISABLE_BACKWARD       / C3_DISABLE_BACKWARD       / backwardFusionEnabled()
 *  - 热路径检测/编译触发       : CT_C3_DISABLE_HOTPATH        / C3_DISABLE_HOTPATH         / hotPathTrackingEnabled()
 *  - MatMul CBLAS 加速         : 运行时 C3_MATMUL_NO_CBLAS=1 / matmulNoCblasEnabled()
 *  - Region 强制合并(跳过代价门) : 运行时 C3_FORCE_REGION_MERGE=1 / forceRegionMergeEnabled()
 *  - planner 影子观测(G2 决策门) : 运行时 C3_PLANNER_SHADOW=1 / plannerShadowEnabled()
 *  - Region 合并策略(ADR-0002)    : 运行时 C3_REGION_MERGE_ALLOW=1 / regionMergeAllowEnabled()
 *  - G3 真接管(planner 参与执行)   : 运行时 C3_G3_TAKEOVER=1 / g3TakeoverEnabled()
 *
 * @date 2026/8/7
 */

#ifndef CTORCH_C3_C3_CONFIG_H
#define CTORCH_C3_C3_CONFIG_H

#include <cstdlib>

namespace ct {
namespace c3 {

namespace detail {

/// 读取 env 布尔开关：值为 "1" 视为开启（返回 true），其余视为关闭
inline bool envFlag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}

} // namespace detail

// ======================= 单 kernel hotpath 注入 =======================
/// 查询单 kernel hotpath 注入是否启用
/// （编译期 CT_C3_DISABLE_SINGLE_KERNEL 宏关闭，或运行时 C3_DISABLE_SINGLE_KERNEL=1，均禁用）
inline bool singleKernelInjectionEnabled() {
#ifdef CT_C3_DISABLE_SINGLE_KERNEL
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_SINGLE_KERNEL");
    return enabled;
#endif
}

// ======================= 区域融合 (Region Fusion) =======================
/// 查询区域融合是否启用
/// （编译期 CT_C3_DISABLE_REGION_FUSION 宏关闭，或运行时 C3_DISABLE_REGION_FUSION=1，均禁用）
inline bool regionFusionEnabled() {
#ifdef CT_C3_DISABLE_REGION_FUSION
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_REGION_FUSION");
    return enabled;
#endif
}

// ======================= 后向融合 (Backward) =======================
/// 查询后向融合是否启用
/// （编译期 CT_C3_DISABLE_BACKWARD 宏关闭，或运行时 C3_DISABLE_BACKWARD=1，均禁用）
inline bool backwardFusionEnabled() {
#ifdef CT_C3_DISABLE_BACKWARD
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_BACKWARD");
    return enabled;
#endif
}

// ======================= 热路径检测/编译触发 =======================
/// 查询热路径检测与编译触发是否启用
/// （编译期 CT_C3_DISABLE_HOTPATH 宏关闭，或运行时 C3_DISABLE_HOTPATH=1，均禁用）
inline bool hotPathTrackingEnabled() {
#ifdef CT_C3_DISABLE_HOTPATH
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_HOTPATH");
    return enabled;
#endif
}

// ======================= MatMul CBLAS 加速 =======================
/// 查询 MatMul 是否禁用 cblas_sgemm（AMX 协处理器）加速
/// （运行时 C3_MATMUL_NO_CBLAS=1 禁用，回退手写标量循环；默认开启 cblas）
inline bool matmulNoCblasEnabled() {
    static const bool disabled = detail::envFlag("C3_MATMUL_NO_CBLAS");
    return disabled;
}

// ======================= Region 融合代价门 (强制合并) =======================
/// 查询是否强制启用 RegionKernel 跨分量合并（跳过代价门，不做收益判定）
/// @details 运行时 C3_FORCE_REGION_MERGE=1 开启。用途：先把"region 划分是否正确"
///          与"划分是否划算"两个正交问题解耦——强制合并用于验证结构等价性
///          (planner 判定 vs MIMO 实际范围)，代价判定作为独立优化层后补。
///          仅影响 FusionPlanner 的分区判定，不改任何 kernel 正确性路径。
inline bool forceRegionMergeEnabled() {
    static const bool enabled = detail::envFlag("C3_FORCE_REGION_MERGE");
    return enabled;
}

// ======================= 影子观测 (G2 迁移决策门) =======================
/// 查询是否启用 planner 影子观测（G2 阶段）
/// @details 运行时 C3_PLANNER_SHADOW=1 开启。语义（区别于 C3_PLANNER_DIAG 的详细诊断）：
///          - **静默观测**：planner 判定与现状 MIMO 一致时不输出（避免日志噪声）
///          - **仅不一致告警**：判错时输出 `[G2-SHADOW-MISMATCH]` 并计入统计, 供事后分析
///          - **绝不改行为**：真实执行仍走 MIMO 手写目录, planner 判定不参与任何决策
///          用途：G2 阶段常态化运行, 累积"planner 会错/不会错"的证据, 为 G3(真接管) 提供依据。
///          默认关闭（保守）；与 C3_PLANNER_DIAG 可共存（后者额外输出详细分区与度量）。
/// @note **2026-09-10 §4.86 起默认开启**：影子纯观测、绝不改行为, 仅在 MIMO 异步编译
///       线程内跑一次 planner 判定(O(节点数), 非热路径)开销可忽略; 常态开启才能持续
///       累积证据。显式设 `C3_PLANNER_SHADOW=0` 可关闭。
inline bool plannerShadowEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("C3_PLANNER_SHADOW");
        if (v == nullptr) return true;   // 默认开(§4.86)
        return v[0] == '1';
    }();
    return enabled;
}

// ======================= Region 合并策略 (ADR-0002) =======================
/// 查询是否启用"跨分量默认合并"策略（ADR-0002 方案 C）
/// @details 运行时 C3_REGION_MERGE_ALLOW=1 开启。语义：跨分量合并默认允许，
///          仅受规模保护上限（`max_region_nodes`）约束，取代现行的相对收益门槛。
///          依据：实测跨分量合并收益仅 0.1% 量级（FFN 5.81µs / 单步 4900µs），
///          该层判别力价值低；判别力下沉到"region 规模保护"。
///          默认关闭 = 现行 Strict 策略（行为完全不变）。
/// @note 若同时设置 C3_FORCE_REGION_MERGE=1，则强制合并优先（跳过一切判定）。
inline bool regionMergeAllowEnabled() {
    static const bool enabled = detail::envFlag("C3_REGION_MERGE_ALLOW");
    return enabled;
}

// ======================= G3 真接管 (planner 参与执行决策) =======================
/// 查询是否启用 G3 真接管：让 planner 判定真正参与 MIMO backward 的融合决策。
/// @details 运行时 `C3_G3_TAKEOVER=0` 可关闭。语义：
///          - 开启后, MIMO backward 编译路径用 FusionPlanner(RegionKernel + Strict)
///            + partitionGraph 切分替代"整图单内核"：planner 判"不合并"则切多内核
///            编排执行, 判"合并"则维持整图单内核。
///          - **默认开启**(2026-09-10 §4.88 起)。依据(本会话实测):
///            ① 数值: FFN 5-step loss / MNIST acc / A/B 设施(切分 vs 整图)均**逐位一致**;
///            ② 性能: 补齐分隔符归属后(§4.87)FC 由"慢 4.17%"转为持平(-0.49%),
///               FFN 32/64/128 三维度均不劣(128 维度 -10%), 即全局持平或更优;
///            ③ 可回退: 显式 `C3_G3_TAKEOVER=0` 即回到整图单内核路径。
///          - 影子观测(C3_PLANNER_SHADOW, 默认开)与实际执行共用同一次 planner 判定,
///            正常情况下恒一致 → 静默; 不一致才告警。
/// @warning 该开关让 planner 判定真正参与执行决策(触碰 backward 执行结构)。
///          首次默认开启时务必跑完回归 + 数值对拍(本会话已完成)。
inline bool g3TakeoverEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("C3_G3_TAKEOVER");
        if (v == nullptr) return true;   // 默认开(§4.88)
        return v[0] == '1';
    }();
    return enabled;
}

// ======================= 分隔符归属 (自动融合可用性) =======================
/// 查询是否允许区域分隔符并入相邻 region（而非各自独立成 LEAF 内核）。
/// @details 运行时 C3_SEPARATOR_MERGE=1 开启。语义：
///          - 分隔符(SumReduce/Softmax/CrossEntropy/Fused/Const)在 region 语义下本是
///            硬边界, 但 region 内核实为"节点间顺序 + 节点内并行", 故可并入。
///          - 独立成 LEAF 的代价 = 多一次内核调用开销; 小图占比高(FC-MIMO 多一个
///            SumReduce 内核 → 慢约 6%), 大图因 region 并行收益大应保持独立。
///          - 并入判据由 RegionFusionPolicy::separator_merge_ws_bytes 给出(工作集上界)。
///          - **默认开启**(2026-09-10 §4.87 起)：该判据经 FC/FFN 实测为纯改进——
///            小图(FC-MIMO ws 80KB/326KB)消除负收益(慢 4.17%/+1.27% → 持平/-0.49%),
///            大图(FFN-MIMO ws 7MB)超出上界、划分不变、不受影响。
///          - 显式设 `C3_SEPARATOR_MERGE=0` 可关闭(回到"分隔符一律独立")。
/// @note 该开关是 G3 接管"全局更优"的关键: 补齐后 FC(该并)与 FFN(该拆)可各得其所。
inline bool separatorMergeEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("C3_SEPARATOR_MERGE");
        if (v == nullptr) return true;   // 默认开(§4.87)
        return v[0] == '1';
    }();
    return enabled;
}

// ======================= 手写 MIMO pattern 退场 (已切换默认, ④ 收口) =======================
/// 查询是否启用**手写** MIMO pattern 的执行段识别(tryExecuteUnifiedMIMOBackward)。
/// @details 手写 pattern(FC MatMul+Add / FFN SwiGLU)曾是 G3 通用接管的"前置识别器"。
///          退场(§4.103-4.107): 通用树式识别器(C3_MIMO_GENERIC)已等价覆盖 FC+FFN,
///          影子对照 + 退场预演逐位一致 → 默认关闭手写执行段(回退通道保留)。
/// @note 默认关闭(手写 pattern 已退场); C3_MIMO_LEGACY=1 可恢复(诊断/回退)。
inline bool mimoLegacyEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("C3_MIMO_LEGACY");
        if (v == nullptr) return false;  // 默认关(手写 pattern 退场, §4.107)
        return v[0] == '1';
    }();
    return enabled;
}

/// 查询是否启用通用树式识别器(手写 MIMO 退场, ADR-012)。
/// @details 通用树捕获(真实拓扑走树 + 通用逐节点反向构建器 + 拓扑缝合 → planner/G3
///          接管)是 FC/FFN 反向的默认路径; miss 时透传给手写识别器(默认已关)。
///          默认开启(§4.107 浸泡后切换); C3_MIMO_GENERIC=0 关闭回退。
/// @note 通用路径与手写路径层叠共存: 通用在前、手写在后, 各自 nullopt 透传。
inline bool mimoGenericEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("C3_MIMO_GENERIC");
        if (v == nullptr) return true;   // 默认开(通用树式识别器为 FC/FFN 默认路径)
        return v[0] == '1';
    }();
    return enabled;
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3_CONFIG_H