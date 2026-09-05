/**
 * @file SIMDTarget.h
 * @brief MLIR 代码生成层的编译期向量宽度（target vector lanes）
 * @details C3 的 MLIR 降层生成 \`vector<N x f32>\` 类型时，向量宽度 N 由宿主机
 *          SIMD 架构在编译期决定（替代原先散落的硬编码 VL=8）：
 *            - AVX-512 (x86_64)  : 16 lanes (512-bit)
 *            - AVX2     (x86_64)  :  8 lanes (256-bit)
 *            - NEON     (aarch64) :  4 lanes (128-bit)
 *            - 其他/标量          :  1 lane
 *          由编译器 -march=native 触发的架构宏自动选择，无需运行期探测。
 *
 * @date 2026/09/05
 */
#ifndef CTORCH_C3_SIMD_TARGET_H
#define CTORCH_C3_SIMD_TARGET_H

#include <cstdint>

namespace ct {
namespace c3 {

/// 编译期确定：MLIR 降层生成 float32 向量时使用的 lane 数
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    inline constexpr std::int64_t kTargetVecLanes = 16;   ///< x86 AVX-512 (F+DQ)
#elif defined(__AVX2__)
    inline constexpr std::int64_t kTargetVecLanes = 8;    ///< x86 AVX2
#elif defined(__aarch64__)
    inline constexpr std::int64_t kTargetVecLanes = 4;    ///< ARM NEON
#else
    inline constexpr std::int64_t kTargetVecLanes = 1;    ///< 标量回退
#endif

/// 人类可读的架构名（调试/日志用）
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    inline constexpr const char* kTargetArchName = "avx512";
#elif defined(__AVX2__)
    inline constexpr const char* kTargetArchName = "avx2";
#elif defined(__aarch64__)
    inline constexpr const char* kTargetArchName = "neon";
#else
    inline constexpr const char* kTargetArchName = "scalar";
#endif

}  // namespace c3
}  // namespace ct

#endif  // CTORCH_C3_SIMD_TARGET_H
