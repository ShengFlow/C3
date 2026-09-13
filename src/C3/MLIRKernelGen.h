/**
 * @file MLIRKernelGen.h
 * @generation JIT-2.x MLIR 标量单算子 IR 后端
 * @note generateFromGraphMLIR 开头内嵌 3.0 路由 tryBuildLinalgElementwise
 *       （命中单节点逐元素则改走 LinalgElementwiseGen），待下轮物理分家时迁出。
 * @brief C3 JIT MLIR kernel 生成器（Phase 1: MLIR/LLVM 后端）
 * @details 将 Graph 编译为 MLIR module，通过标准 lowering pipeline 降至 LLVM IR，
 *          再经 ExecutionEngine JIT 编译为原生函数指针。替代 HandwrittenKernelGen。
 *          输出与 HandwrittenKernelGen 完全相同的 GeneratedKernel 结构体，
 *          下游 C3KernelRegistry / CtorchScheduler 无需任何改动。
 *
 *          Pipeline: Graph → MLIR (arith+scf+memref+func) → LLVM dialect → JIT
 * @date 2026/8/1
 *
 * [Dev] v0.5.2 DCU 接入 refactor (2026-08-10):
 *   抽 buildMLIRModule + applyLoweringPipeline 公开 API, 让 MLIRToLLVMIR.cpp
 *   的 mlirToLLVMIRFromGraph 复用同一份 build / lower 逻辑 (不重复代码).
 */

#ifndef CTORCH_C3_MLIR_KERNEL_GEN_H
#define CTORCH_C3_MLIR_KERNEL_GEN_H

#include "C3/GeneratedKernel.h"  // 复用 GeneratedKernel 定义 (SHARED)
#include <mutex>

namespace mlir {
    class MLIRContext;
    class ModuleOp;
    class PassManager;
    template <typename T> class OwningOpRef;
}

namespace ct {
namespace c3 {

extern std::mutex c3_global_mlir_mutex;

/**
 * @brief 从 Graph 生成 MLIR 编译的 kernel（Phase 1 LLVM 后端）
 * @param graph 经过 canonicalize 的计算图
 * @param opt_level LLVM 优化级别（0=O0, 1=O1, 2=O2, 3=O3/Ofast，默认 2）
 * @return GeneratedKernel 包含函数指针和资源管理回调
 * @throw std::runtime_error 编译失败时抛出
 */
GeneratedKernel generateFromGraphMLIR(const Graph& graph, int opt_level = 2);

/**
 * @brief [v0.5.2 公开] 从 C3 Graph 构建 MLIR Module
 * @details 之前是 file-static, 现抽公开让 MLIRToLLVMIR.cpp 复用
 *
 * @param context 已注册必要 dialect (arith/math/scf/func/memref/LLVM) 的 MLIRContext
 * @param graph C3 Graph
 * @return OwningOpRef<ModuleOp>
 * @throw std::runtime_error 当 graph 校验失败
 */
mlir::OwningOpRef<mlir::ModuleOp> buildMLIRModule(mlir::MLIRContext& context, const Graph& graph,
                                        size_t* out_pool_buf_count = nullptr);

/**
 * @brief [v0.5.2 公开] 对 MLIR Module 跑标准 lowering pipeline
 * @details 顺序(opt_level>=4 时在 SCFToCF 前追加 ControlFlowSink/RemoveDeadValues):
 *          StripDebugInfo → Canonicalizer → runC3Combine(高层图优化) →
 *          runC3Lowering(C3 算子 → 标量/向量循环) → CSE → SymbolDCE → LICM →
 *          SCFForLoopCanonicalization → SCFToCF → MathToLLVM → ArithToLLVM →
 *          CFToLLVM → FuncToLLVM → MemRefToLLVM → ReconcileUnrealizedCasts →
 *          Canonicalizer → CSE
 *          跑完 module 在 LLVM dialect, 可直接喂 mlir::translateModuleToLLVMIR
 * @note 原文档仅列 "Canonicalizer → CSE → LICM → SCFToCF → ArithToLLVM → ...",
 *       漏记 MathToLLVM 与 runC3Combine/runC3Lowering 两个高层阶段(实现在
 *       C3DialectLowering.cpp), 已于 §4.112 更正。
 */
void applyLoweringPipeline(mlir::ModuleOp module, int opt_level = 3);

/**
 * @brief [§4.112] 追加「MLIR → LLVM」公共 lowering 尾段(linalg codegen 三条路径共用)
 * @details 尾段固定为 9 个 pass:
 *          SCFToCF → ArithToLLVM → MathToLLVM → CFToLLVM → FuncToLLVM →
 *          MemRefToLLVM → ReconcileUnrealizedCasts → Canonicalizer → CSE
 *          此前 LinalgElementwiseGen / LinalgFusedGen / LinalgOneShotGen 各内联一份,
 *          三份逐字相同 —— 抽为单一入口, 避免只改一处即产生流水线分歧
 *          (典型失效模式: 漏 ReconcileUnrealizedCasts 导致 unrealized_conversion_cast 残留)。
 * @note 只做 addPass, **不创建也不运行** PassManager: 调用方保留自己的分阶段
 *       PassManager 与「哪一段失败」的错误定位。
 */
void appendLLVMLoweringTail(mlir::PassManager& pm);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_MLIR_KERNEL_GEN_H