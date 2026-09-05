/**
 * @file C3DialectLowering.cpp
 * @generation JIT-3.0 C3 Dialect 算子降低与优化管线
 * @brief JIT 3.0 C3 Dialect 算子到标准 Linalg/SCF 的 Lowering 与优化管线（CGO 2027 解耦版）
 * @details 物理隔离 JIT 3.0 的高层降低转换模式，彻底拆分编译逻辑。
 *          通过 RewritePattern 将 c3.matmul、c3.transpose、c3.sum_reduce 算子
 *          完美降解到 linalg / scf / math / arith 层级，并应用自动融合与 One-Shot 缓冲化。
 * @date 2026/08/16
 */

#include "MLIRKernelGen.h"
#include "C3/C3Config.h"
#include "C3/SIMDTarget.h"
#include "C3/C3Dialect.h"
#include "C3/TuningState.h"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

// 导入 TableGen 自动生成的 C3Combine (DRR 规则) 模式重写实现
#include "C3Combine.cpp.inc"

namespace ct {
namespace c3 {

namespace {

// ======================= Helper for Tiling / GEP =======================

static mlir::Value indexToI64(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value idx) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), idx);
}

static mlir::Value i64ToIndex(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

static void buildLoop(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value n, int64_t known_numel,
                      const std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value)>& body_fn) {
    if (known_numel > 0 && known_numel <= 16) {
        for (int64_t i = 0; i < known_numel; ++i) {
            mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
            mlir::Value idx_i64 = indexToI64(builder, loc, idx);
            body_fn(builder, loc, idx_i64);
        }
    } else {
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
        mlir::Value n_idx = i64ToIndex(builder, loc, n);
        auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
        builder.setInsertionPointToStart(loop.getBody());
        mlir::Value idx = loop.getInductionVar();
        mlir::Value idx_i64 = indexToI64(builder, loc, idx);
        body_fn(builder, loc, idx_i64);
        builder.setInsertionPointAfter(loop);
    }
}

static void buildVectorizedLoop(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value n, int64_t known_numel,
                                const std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value)>& vec_body_fn,
                                const std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value)>& scalar_body_fn) {
    constexpr int64_t VL = ct::c3::kTargetVecLanes;
    auto f32 = builder.getF32Type();

    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value VL_v = builder.create<mlir::arith::ConstantIndexOp>(loc, VL);

    mlir::Value n_idx = i64ToIndex(builder, loc, n);
    mlir::Value rem = builder.create<mlir::arith::RemUIOp>(loc, n_idx, VL_v);
    mlir::Value n_vec = builder.create<mlir::arith::SubIOp>(loc, n_idx, rem);

    // Vector Loop
    auto vloop = builder.create<mlir::scf::ForOp>(loc, c0, n_vec, VL_v);
    builder.setInsertionPointToStart(vloop.getBody());
    mlir::Value base = indexToI64(builder, loc, vloop.getInductionVar());
    vec_body_fn(builder, loc, base);
    builder.setInsertionPointAfter(vloop);

    // Scalar Loop (Cleanup)
    auto sloop = builder.create<mlir::scf::ForOp>(loc, n_vec, n_idx, c1);
    builder.setInsertionPointToStart(sloop.getBody());
    mlir::Value idx = indexToI64(builder, loc, sloop.getInductionVar());
    scalar_body_fn(builder, loc, idx);
    builder.setInsertionPointAfter(sloop);
}

// [Fix 2026-09-05 苏璃珞] 标量广播成 <VL x f32> 向量。
// 仓库不做 vector 方言（vector.broadcast 在 LLVM translation 上会触发 VectorToLLVM
// greedy 不收敛 / missing LLVMTranslationDialectInterface）。与 MLIRKernelGen 595-607
// 的标量→vector 展开保持一致：undef + VL 次 insertelement。
static mlir::Value buildScalarSplatVec(mlir::OpBuilder& builder, mlir::Location loc,
                                       mlir::Value scalar, mlir::Type vec_ty) {
    constexpr int64_t VL = ct::c3::kTargetVecLanes;
    mlir::Value vec = builder.create<mlir::LLVM::UndefOp>(loc, vec_ty);
    for (int64_t lane = 0; lane < VL; ++lane) {
        mlir::Value lane_idx = builder.create<mlir::arith::ConstantIntOp>(loc, lane, 32);
        vec = builder.create<mlir::LLVM::InsertElementOp>(loc, vec_ty, vec, scalar, lane_idx);
    }
    return vec;
}

static void buildSmallMatMul(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value lhs, mlir::Value rhs, mlir::Value out, mlir::Value bias,
                             mlir::Value preAct,   // [Prewalk A] 可选 pre-activation 输出（可为 nullptr）
                             size_t M, size_t K, size_t N,
                             int transA, int transB, int act,
                             size_t bias_numel) {
    auto f32 = builder.getF32Type();
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::Value c1_i64 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);

    mlir::Value M_v = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
    mlir::Value N_v = builder.create<mlir::arith::ConstantIndexOp>(loc, N);
    mlir::Value K_v = builder.create<mlir::arith::ConstantIndexOp>(loc, K);

    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto loop_i = builder.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
    builder.setInsertionPointToStart(loop_i.getBody());
    mlir::Value i_idx = loop_i.getInductionVar();
    mlir::Value i_i64 = indexToI64(builder, loc, i_idx);

    auto loop_j = builder.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
    builder.setInsertionPointToStart(loop_j.getBody());
    mlir::Value j_idx = loop_j.getInductionVar();
    mlir::Value j_i64 = indexToI64(builder, loc, j_idx);

    mlir::Value out_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, builder.create<mlir::arith::ConstantIntOp>(loc, N, 64));
    out_idx = builder.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
    mlir::Value out_cell_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});

    mlir::Value init_val = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
    if (bias) {
        mlir::Value bias_idx = j_i64;
        if (bias_numel == M) {
            bias_idx = i_i64;
        } else if (bias_numel == 1) {
            bias_idx = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
        }
        mlir::Value bias_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{bias_idx});
        init_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
    }

    auto loop_k = builder.create<mlir::scf::ForOp>(loc, c0, K_v, c1, mlir::ValueRange{init_val});
    builder.setInsertionPointToStart(loop_k.getBody());
    mlir::Value k_idx = loop_k.getInductionVar();
    mlir::Value k_i64 = indexToI64(builder, loc, k_idx);
    mlir::Value sum_accum = loop_k.getRegionIterArgs()[0];

    mlir::Value a_idx;
    if (transA == 112) {
        a_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, builder.create<mlir::arith::ConstantIntOp>(loc, M, 64));
        a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, i_i64);
    } else {
        a_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, builder.create<mlir::arith::ConstantIntOp>(loc, K, 64));
        a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, k_i64);
    }

    mlir::Value b_idx;
    if (transB == 112) {
        b_idx = builder.create<mlir::arith::MulIOp>(loc, j_i64, builder.create<mlir::arith::ConstantIntOp>(loc, K, 64));
        b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, k_i64);
    } else {
        b_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, builder.create<mlir::arith::ConstantIntOp>(loc, N, 64));
        b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, j_i64);
    }

    mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{a_idx});
    mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{b_idx});

    mlir::Value av = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
    mlir::Value bv = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);
    mlir::Value prod = builder.create<mlir::arith::MulFOp>(loc, av, bv);
    mlir::Value next_sum = builder.create<mlir::arith::AddFOp>(loc, sum_accum, prod);

    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{next_sum});

    builder.setInsertionPointAfter(loop_k);
    mlir::Value final_sum = loop_k.getResult(0);

    // [Prewalk A] 需要时把 pre-activation 值写一份到 preAct（backward 复用中间值）
    if (preAct) {
        mlir::Value preAct_cell_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, preAct, mlir::ValueRange{out_idx});
        builder.create<mlir::LLVM::StoreOp>(loc, final_sum, preAct_cell_ptr);
    }

    mlir::Value activated = final_sum;
    if (act == 1) { // ReLU
        mlir::Value zero = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
        activated = builder.create<mlir::arith::MaxNumFOp>(loc, final_sum, zero);
    } else if (act == 2) { // Sigmoid
        mlir::Value neg_sum = builder.create<mlir::arith::NegFOp>(loc, final_sum);
        mlir::Value exp_val = builder.create<mlir::math::ExpOp>(loc, neg_sum);
        mlir::Value one = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));
        mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, one, exp_val);
        activated = builder.create<mlir::arith::DivFOp>(loc, one, denom);
    } else if (act == 3) { // Tanh
        activated = builder.create<mlir::math::TanhOp>(loc, final_sum);
    }

    builder.create<mlir::LLVM::StoreOp>(loc, activated, out_cell_ptr);

    builder.setInsertionPointAfter(loop_j);
    builder.setInsertionPointAfter(loop_i);
}

} // namespace

// ======================= JIT 3.0 C3 Dialect Lowering Patterns =======================

struct TransposeOpLowering : public mlir::OpRewritePattern<mlir::c3::TransposeOp> {
    using OpRewritePattern<mlir::c3::TransposeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::TransposeOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        int64_t M = op.getM();
        int64_t N = op.getN();
        int dim0 = op.getDim0();
        int dim1 = op.getDim1();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        if ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) {
            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto loop_i = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
            rewriter.setInsertionPointToStart(loop_i.getBody());
            mlir::Value i_idx = loop_i.getInductionVar();
            mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);

            auto loop_j = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
            rewriter.setInsertionPointToStart(loop_j.getBody());
            mlir::Value j_idx = loop_j.getInductionVar();
            mlir::Value j_i64 = indexToI64(rewriter, loc, j_idx);

            mlir::Value in_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
            in_idx = rewriter.create<mlir::arith::AddIOp>(loc, in_idx, j_i64);

            mlir::Value out_idx = rewriter.create<mlir::arith::MulIOp>(loc, j_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, M, 64));
            out_idx = rewriter.create<mlir::arith::AddIOp>(loc, out_idx, i_i64);

            mlir::Value in_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{in_idx});
            mlir::Value out_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});

            mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            rewriter.create<mlir::LLVM::StoreOp>(loc, val, out_ptr);

            rewriter.setInsertionPointAfter(loop_j);
            rewriter.setInsertionPointAfter(loop_i);
        } else {
            int64_t total_ops = M * N;
            buildLoop(rewriter, loc, rewriter.create<mlir::arith::ConstantIntOp>(loc, total_ops, 64), total_ops,
                [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                    mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                    mlir::Value out_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});
                    mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                    bld.create<mlir::LLVM::StoreOp>(loc, val, out_ptr);
                });
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

template <typename SrcOp, typename ArithOp>
struct BinaryOpLowering : public mlir::OpRewritePattern<SrcOp> {
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();
        mlir::Value out = op.getOut();
        int64_t bmod = op.getBmod();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        if (bmod == 0) {
            constexpr int64_t VL = ct::c3::kTargetVecLanes;
            auto vec_ty = mlir::VectorType::get({VL}, f32);

            buildVectorizedLoop(rewriter, loc, op.getNumel(), 0,
                [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value base) {
                    mlir::Value l_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{base});
                    mlir::Value r_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{base});
                    mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{base});

                    mlir::Value lv = bld.create<mlir::LLVM::LoadOp>(loc, vec_ty, l_ptr, 16);
                    mlir::Value rv = bld.create<mlir::LLVM::LoadOp>(loc, vec_ty, r_ptr, 16);
                    mlir::Value res = bld.create<ArithOp>(loc, lv, rv);
                    bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
                },
                [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                    mlir::Value l_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{idx});
                    mlir::Value r_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{idx});
                    mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                    mlir::Value lv = bld.create<mlir::LLVM::LoadOp>(loc, f32, l_ptr);
                    mlir::Value rv = bld.create<mlir::LLVM::LoadOp>(loc, f32, r_ptr);
                    mlir::Value res = bld.create<ArithOp>(loc, lv, rv);
                    bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
                });
        } else {
            buildLoop(rewriter, loc, op.getNumel(), 0,
                [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                    mlir::Value l_idx = idx;
                    mlir::Value r_idx = idx;
                    if (bmod > 0) {
                        mlir::Value mod_val = bld.create<mlir::arith::ConstantIntOp>(loc, bmod, 64);
                        r_idx = bld.create<mlir::arith::RemUIOp>(loc, idx, mod_val);
                    } else if (bmod < 0) {
                        mlir::Value mod_val = bld.create<mlir::arith::ConstantIntOp>(loc, -bmod, 64);
                        l_idx = bld.create<mlir::arith::RemUIOp>(loc, idx, mod_val);
                    }
                    mlir::Value l_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{l_idx});
                    mlir::Value r_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{r_idx});
                    mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                    mlir::Value lv = bld.create<mlir::LLVM::LoadOp>(loc, f32, l_ptr);
                    mlir::Value rv = bld.create<mlir::LLVM::LoadOp>(loc, f32, r_ptr);
                    mlir::Value res = bld.create<ArithOp>(loc, lv, rv);
                    bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
                });
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

using AddOpLowering = BinaryOpLowering<mlir::c3::AddOp, mlir::arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<mlir::c3::SubOp, mlir::arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<mlir::c3::MulOp, mlir::arith::MulFOp>;
using DivOpLowering = BinaryOpLowering<mlir::c3::DivOp, mlir::arith::DivFOp>;

template <typename SrcOp, typename ArithOp>
struct UnaryOpLowering : public mlir::OpRewritePattern<SrcOp> {
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        constexpr int64_t VL = ct::c3::kTargetVecLanes;
        auto vec_ty = mlir::VectorType::get({VL}, f32);

        buildVectorizedLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value base) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{base});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{base});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, vec_ty, in_ptr, 16);
                mlir::Value res = bld.create<ArithOp>(loc, val);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
            },
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                mlir::Value res = bld.create<ArithOp>(loc, val);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

using NegOpLowering     = UnaryOpLowering<mlir::c3::NegOp, mlir::arith::NegFOp>;
using TanhOpLowering    = UnaryOpLowering<mlir::c3::TanhOp, mlir::math::TanhOp>;
using ExpOpLowering     = UnaryOpLowering<mlir::c3::ExpOp, mlir::math::ExpOp>;
using LogOpLowering     = UnaryOpLowering<mlir::c3::LogOp, mlir::math::LogOp>;

struct ReLUOpLowering : public mlir::OpRewritePattern<mlir::c3::ReLUOp> {
    using OpRewritePattern<mlir::c3::ReLUOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::ReLUOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        constexpr int64_t VL = ct::c3::kTargetVecLanes;
        auto vec_ty = mlir::VectorType::get({VL}, f32);

        buildVectorizedLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value base) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{base});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{base});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, vec_ty, in_ptr, 16);
                mlir::Value zero = bld.create<mlir::arith::ConstantOp>(
                    loc, mlir::DenseElementsAttr::get(vec_ty, 0.0f));
                mlir::Value res = bld.create<mlir::arith::MaxNumFOp>(loc, val, zero);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
            },
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                mlir::Value zero = bld.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
                mlir::Value res = bld.create<mlir::arith::MaxNumFOp>(loc, val, zero);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct SigmoidOpLowering : public mlir::OpRewritePattern<mlir::c3::SigmoidOp> {
    using OpRewritePattern<mlir::c3::SigmoidOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::SigmoidOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        constexpr int64_t VL = ct::c3::kTargetVecLanes;
        auto vec_ty = mlir::VectorType::get({VL}, f32);

        buildVectorizedLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value base) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{base});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{base});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, vec_ty, in_ptr, 16);
                mlir::Value neg_x = bld.create<mlir::arith::NegFOp>(loc, val);
                mlir::Value exp_val = bld.create<mlir::math::ExpOp>(loc, neg_x);
                mlir::Value one = bld.create<mlir::arith::ConstantOp>(
                    loc, mlir::DenseElementsAttr::get(vec_ty, 1.0f));
                mlir::Value denom = bld.create<mlir::arith::AddFOp>(loc, one, exp_val);
                mlir::Value res = bld.create<mlir::arith::DivFOp>(loc, one, denom);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
            },
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                mlir::Value neg_x = bld.create<mlir::arith::NegFOp>(loc, val);
                mlir::Value exp_val = bld.create<mlir::math::ExpOp>(loc, neg_x);
                mlir::Value one = bld.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));
                mlir::Value denom = bld.create<mlir::arith::AddFOp>(loc, one, exp_val);
                mlir::Value res = bld.create<mlir::arith::DivFOp>(loc, one, denom);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr, 16);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct SumReduceOpLowering : public mlir::OpRewritePattern<mlir::c3::SumReduceOp> {
    using OpRewritePattern<mlir::c3::SumReduceOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::SumReduceOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        int64_t M = op.getM();
        int64_t N = op.getN();
        int axis = op.getAxis();
        // [P0.2 2026-08-30 苏璃珞] keepdim 仅影响 shape 描述（[M] vs [M,1]），
        // 输出 buffer 元素数不变，所以现有 axis=0/1 两个分支的循环体不需要改。
        // 这里取出来只用于未来可能的 metadata 透传。
        (void)op.getKeepdim();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        if (axis == 0) {
            // [MIMO epilogue 优化] 原实现「外 j 内 i」：input[i*N+j] 沿 i 跨 N*4=1KB stride，
            // cache 极差 + 标量。改为「外 i 内 j」行优先：input 行连续读、out 连续累积，
            // LLVM 可向量化内层 for-j。同一列 j 的累加仍按 i 升序，浮点次序不变，数值等价。
            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
            auto mk_zero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));

            // 1) 清零 out[0,N)
            {
                auto lzero = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
                rewriter.setInsertionPointToStart(lzero.getBody());
                auto jo = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out,
                    mlir::ValueRange{indexToI64(rewriter, loc, lzero.getInductionVar())});
                rewriter.create<mlir::LLVM::StoreOp>(loc, mk_zero, jo);
                rewriter.setInsertionPointAfter(lzero);
            }
            // 2) 行优先累积: 外 i(0,M), 内 j(0,N) —— input 行连续读, out 连续 load/store
            {
                auto lrow = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
                rewriter.setInsertionPointToStart(lrow.getBody());
                auto iN = rewriter.create<mlir::arith::MulIOp>(loc,
                    indexToI64(rewriter, loc, lrow.getInductionVar()),
                    rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
                auto lcol = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
                rewriter.setInsertionPointToStart(lcol.getBody());
                auto j_idx = indexToI64(rewriter, loc, lcol.getInductionVar());
                auto in_idx = rewriter.create<mlir::arith::AddIOp>(loc, iN, j_idx);
                auto in_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{in_idx});
                auto out_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{j_idx});
                auto val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                auto old = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, out_ptr);
                rewriter.create<mlir::LLVM::StoreOp>(loc, rewriter.create<mlir::arith::AddFOp>(loc, old, val), out_ptr);
                rewriter.setInsertionPointAfter(lcol);
                rewriter.setInsertionPointAfter(lrow);
            }
        } else {
            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto loop_i = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
            rewriter.setInsertionPointToStart(loop_i.getBody());
            mlir::Value i_idx = loop_i.getInductionVar();
            mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);

            mlir::Value out_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{i_i64});
            mlir::Value zero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
            rewriter.create<mlir::LLVM::StoreOp>(loc, zero, out_ptr);

            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            auto loop_j = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
            rewriter.setInsertionPointToStart(loop_j.getBody());
            mlir::Value j_idx = loop_j.getInductionVar();
            mlir::Value j_i64 = indexToI64(rewriter, loc, j_idx);

            mlir::Value in_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
            in_idx = rewriter.create<mlir::arith::AddIOp>(loc, in_idx, j_i64);

            mlir::Value in_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{in_idx});
            mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            mlir::Value old_sum = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, out_ptr);
            mlir::Value new_sum = rewriter.create<mlir::arith::AddFOp>(loc, old_sum, val);
            rewriter.create<mlir::LLVM::StoreOp>(loc, new_sum, out_ptr);

            rewriter.setInsertionPointAfter(loop_j);
            rewriter.setInsertionPointAfter(loop_i);
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

static mlir::LLVM::LLVMFuncOp getOrDeclareCblasSgemm(mlir::OpBuilder& builder, mlir::Location loc) {
    auto* ctx = builder.getContext();
    auto module_op = builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (!module_op)
        throw std::runtime_error("getOrDeclareCblasSgemm: not inside a module");
    auto existing = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>("cblas_sgemm");
    if (existing) return existing;

    auto i32 = mlir::IntegerType::get(ctx, 32);
    auto f32 = mlir::Float32Type::get(ctx);
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(ctx);

    // void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
    auto sgemm_type = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(ctx),
        {i32, i32, i32, i32, i32, i32, f32, ptr_type, i32, ptr_type, i32, f32, ptr_type, i32},
        false);

    auto saved_ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module_op.getBody());
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "cblas_sgemm", sgemm_type);
    func.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(saved_ip);
    return func;
}

struct MatMulOpLowering : public mlir::OpRewritePattern<mlir::c3::MatMulOp> {
    using OpRewritePattern<mlir::c3::MatMulOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::MatMulOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();
        mlir::Value out = op.getOut();
        mlir::Value bias = op.getBias();
        mlir::Value preAct = op.getPreAct();   // [Prewalk A] 可选 pre-activation 输出（可为 nullptr）

        int64_t M = op.getM();
        int64_t K = op.getK();
        int64_t N = op.getN();
        int transA = op.getTransA();
        int transB = op.getTransB();
        int act = op.getAct();
        int64_t tileM = op.getTileM();
        int64_t tileN = op.getTileN();
        int64_t bias_numel = op.getBiasNumel();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        bool fallback_to_small = false;
        int64_t total_ops = M * N * K;
        // C3_MATMUL_NO_CBLAS=1 时禁用 cblas_sgemm（AMX）加速，整体回退手写标量循环（逃生开关）
        if (total_ops < 256 || ct::c3::matmulNoCblasEnabled()) {
            fallback_to_small = true;
        } else {
            // [CBLAS sgemm branch]
            auto i32 = rewriter.getI32Type();
            mlir::Value val_order = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(101)); // CblasRowMajor
            mlir::Value val_transA = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(transA == 112 ? 112 : 111));
            mlir::Value val_transB = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(transB == 112 ? 112 : 111));
            mlir::Value val_M = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr((int32_t)M));
            mlir::Value val_N = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr((int32_t)N));
            mlir::Value val_K = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr((int32_t)K));
            mlir::Value val_alpha = rewriter.create<mlir::LLVM::ConstantOp>(loc, f32, rewriter.getF32FloatAttr(1.0f));
            
            int32_t lda_v = (transA == 112) ? (int32_t)M : (int32_t)K;
            int32_t ldb_v = (transB == 112) ? (int32_t)K : (int32_t)N;
            int32_t ldc_v = (int32_t)N;

            mlir::Value val_lda = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(lda_v));
            mlir::Value val_ldb = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(ldb_v));
            mlir::Value val_beta = rewriter.create<mlir::LLVM::ConstantOp>(loc, f32, rewriter.getF32FloatAttr(0.0f));
            mlir::Value val_ldc = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(ldc_v));

            auto sgemm_func = getOrDeclareCblasSgemm(rewriter, loc);
            rewriter.create<mlir::LLVM::CallOp>(loc, sgemm_func, mlir::ValueRange{
                val_order, val_transA, val_transB, val_M, val_N, val_K,
                val_alpha, lhs, val_lda, rhs, val_ldb, val_beta, out, val_ldc
            });

            // Epilogue for bias and activation — row-wise vectorized (VL = kTargetVecLanes)
            // 外层仍是 M 行的 scf.for；每行 N 个连续元素改走 buildVectorizedLoop：
            //   vector body: LD out/bias (VL-wide), ADD bias (if needed), MAX(ZERO)/exp/tanh, ST 回去 —— 与 BinaryOpLowering / ReLUOpLowering 完全一致的对齐策略
            if (bias || act != 0) {
                auto& b = rewriter;
                constexpr int64_t VL = ct::c3::kTargetVecLanes;
                auto vec_ty = mlir::VectorType::get({VL}, f32);
                mlir::Value M_v = b.create<mlir::arith::ConstantIndexOp>(loc, M);
                mlir::Value c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
                mlir::Value c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
                mlir::Value N_i64 = b.create<mlir::arith::ConstantIntOp>(loc, N, 64);

                auto loop_i = b.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
                b.setInsertionPointToStart(loop_i.getBody());
                mlir::Value i_idx = loop_i.getInductionVar();
                mlir::Value i_i64 = indexToI64(b, loc, i_idx);
                mlir::Value row_base = b.create<mlir::arith::MulIOp>(loc, i_i64, N_i64);

                // 仅当 bias_numel == M (按行广播) 时才需要 row-specific bias_idx
                // bias_numel == N 时每行 bias 都是 [0..N)，bias_numel == 1 都是 0，可在 inner 内直接算
                mlir::Value bias_row_idx;
                if (bias && bias_numel == M) bias_row_idx = i_i64;

                auto vec_body = [&](mlir::OpBuilder& vb, mlir::Location vloc, mlir::Value col_base_i64) {
                    mlir::Value out_row_off = vb.create<mlir::arith::AddIOp>(vloc, row_base, col_base_i64);
                    mlir::Value out_ptr = vb.create<mlir::LLVM::GEPOp>(vloc, ptr_type, f32, out, mlir::ValueRange{out_row_off});
                    mlir::Value val_vec = vb.create<mlir::LLVM::LoadOp>(vloc, vec_ty, out_ptr, /*alignment=*/16);

                    if (bias) {
                        mlir::Value b_vec;
                        if (bias_numel == M) {
                            // row broadcast: 同一块 bias_val 广播 VL 次（undef+insertelement splat, 与 MLIRKernelGen 一致）
                            mlir::Value bias_ptr = vb.create<mlir::LLVM::GEPOp>(vloc, ptr_type, f32, bias, mlir::ValueRange{bias_row_idx});
                            mlir::Value b_s = vb.create<mlir::LLVM::LoadOp>(vloc, f32, bias_ptr);
                            b_vec = buildScalarSplatVec(vb, vloc, b_s, vec_ty);
                        } else if (bias_numel == 1) {
                            mlir::Value bias_0 = vb.create<mlir::arith::ConstantIntOp>(vloc, 0, 64);
                            mlir::Value bias_ptr = vb.create<mlir::LLVM::GEPOp>(vloc, ptr_type, f32, bias, mlir::ValueRange{bias_0});
                            mlir::Value b_s = vb.create<mlir::LLVM::LoadOp>(vloc, f32, bias_ptr);
                            b_vec = buildScalarSplatVec(vb, vloc, b_s, vec_ty);
                        } else {
                            // bias_numel == N (按列广播，最常见): bias[col..col+VL-1] 连续 VL 个 LD
                            mlir::Value bias_ptr = vb.create<mlir::LLVM::GEPOp>(vloc, ptr_type, f32, bias, mlir::ValueRange{col_base_i64});
                            b_vec = vb.create<mlir::LLVM::LoadOp>(vloc, vec_ty, bias_ptr, /*alignment=*/16);
                        }
                        val_vec = vb.create<mlir::arith::AddFOp>(vloc, val_vec, b_vec);
                    }

                    // [Prewalk A] 需要时把 pre-activation 值同时写一份到 preAct（backward 复用中间值）
                    if (preAct) {
                        mlir::Value preAct_ptr = vb.create<mlir::LLVM::GEPOp>(vloc, ptr_type, f32, preAct, mlir::ValueRange{out_row_off});
                        vb.create<mlir::LLVM::StoreOp>(vloc, val_vec, preAct_ptr, /*alignment=*/16);
                    }

                    mlir::Value act_vec = val_vec;
                    if (act == 1) { // ReLU (vector.max with zero splat)
                        // [Fix 2026-09-05] 用 DenseElementsAttr splat 构造 vector 常数，
                        // 与 ReLUOpLowering 一致，避免 vector.broadcast 触发 VectorToLLVM 不收敛
                        mlir::Value zero_v = vb.create<mlir::arith::ConstantOp>(
                            vloc, mlir::DenseElementsAttr::get(vec_ty, 0.0f));
                        act_vec = vb.create<mlir::arith::MaxNumFOp>(vloc, val_vec, zero_v);
                    } else if (act == 2) { // Sigmoid: 1 / (1 + exp(-x))
                        mlir::Value neg = vb.create<mlir::arith::NegFOp>(vloc, val_vec);
                        mlir::Value e = vb.create<mlir::math::ExpOp>(vloc, neg);
                        mlir::Value one_v = vb.create<mlir::arith::ConstantOp>(
                            vloc, mlir::DenseElementsAttr::get(vec_ty, 1.0f));
                        mlir::Value d = vb.create<mlir::arith::AddFOp>(vloc, one_v, e);
                        act_vec = vb.create<mlir::arith::DivFOp>(vloc, one_v, d);
                    } else if (act == 3) { // Tanh
                        act_vec = vb.create<mlir::math::TanhOp>(vloc, val_vec);
                    }

                    vb.create<mlir::LLVM::StoreOp>(vloc, act_vec, out_ptr, /*alignment=*/16);
                };

                auto scalar_body = [&](mlir::OpBuilder& sb, mlir::Location sloc, mlir::Value j_i64) {
                    mlir::Value out_idx = sb.create<mlir::arith::AddIOp>(sloc, row_base, j_i64);
                    mlir::Value out_cell_ptr = sb.create<mlir::LLVM::GEPOp>(sloc, ptr_type, f32, out, mlir::ValueRange{out_idx});
                    mlir::Value val = sb.create<mlir::LLVM::LoadOp>(sloc, f32, out_cell_ptr);
                    if (bias) {
                        mlir::Value bias_idx = j_i64;
                        if (bias_numel == M) bias_idx = bias_row_idx;
                        else if (bias_numel == 1) bias_idx = sb.create<mlir::arith::ConstantIntOp>(sloc, 0, 64);
                        mlir::Value bias_ptr = sb.create<mlir::LLVM::GEPOp>(sloc, ptr_type, f32, bias, mlir::ValueRange{bias_idx});
                        mlir::Value bias_val = sb.create<mlir::LLVM::LoadOp>(sloc, f32, bias_ptr);
                        val = sb.create<mlir::arith::AddFOp>(sloc, val, bias_val);
                    }
                    if (preAct) {
                        mlir::Value preAct_cell_ptr = sb.create<mlir::LLVM::GEPOp>(sloc, ptr_type, f32, preAct, mlir::ValueRange{out_idx});
                        sb.create<mlir::LLVM::StoreOp>(sloc, val, preAct_cell_ptr);
                    }
                    mlir::Value activated = val;
                    if (act == 1) {
                        mlir::Value zero = sb.create<mlir::arith::ConstantFloatOp>(sloc, f32, llvm::APFloat(0.0f));
                        activated = sb.create<mlir::arith::MaxNumFOp>(sloc, val, zero);
                    } else if (act == 2) {
                        mlir::Value neg_sum = sb.create<mlir::arith::NegFOp>(sloc, val);
                        mlir::Value exp_val = sb.create<mlir::math::ExpOp>(sloc, neg_sum);
                        mlir::Value one = sb.create<mlir::arith::ConstantFloatOp>(sloc, f32, llvm::APFloat(1.0f));
                        mlir::Value denom = sb.create<mlir::arith::AddFOp>(sloc, one, exp_val);
                        activated = sb.create<mlir::arith::DivFOp>(sloc, one, denom);
                    } else if (act == 3) {
                        activated = sb.create<mlir::math::TanhOp>(sloc, val);
                    }
                    sb.create<mlir::LLVM::StoreOp>(sloc, activated, out_cell_ptr);
                };

                // [Fix 2026-09-05 苏璃珞] MatMul epilogue 向量化：vec_body 实际执行。
                // 常数用 DenseElementsAttr splat、row/标量 bias 用 undef+insertelement
                // splat(buildScalarSplatVec)、列 bias 走 vector load —— 全路径无 vector 方言 op,
                // 无需 VectorToLLVM,不触发 greedy 不收敛 / missing translation。
                mlir::Value N_val = N_i64;
                buildVectorizedLoop(b, loc, N_val, (int64_t)N, vec_body, scalar_body);

                b.setInsertionPointAfter(loop_i);
            }
        }

        if (fallback_to_small) {
            buildSmallMatMul(rewriter, loc, lhs, rhs, out, bias, preAct,
                             (size_t)M, (size_t)K, (size_t)N,
                             transA, transB, act, (size_t)bias_numel);
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

static void runC3Combine(mlir::ModuleOp module) {
    mlir::RewritePatternSet patterns(module.getContext());
    populateWithGenerated(patterns);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
        throw std::runtime_error("C3DialectLowering: C3Combine pattern rewrite optimization failed");
    }
}

// [P0.2 2026-08-30 苏璃珞] SoftmaxOpLowering：c3.softmax → linalg.softmax（MLIR 标准 op，底层向量化）
//
// 关键设计：直接用 mlir::linalg::SoftmaxOp（MLIR upstream 标准 op，**内部已实现** rowmax → exp → rowsum → div 4 步）。
//   - 数值稳定：linalg.softmax 内部用 max-subtraction 防 exp 溢出
//   - **向量化**：`convert-linalg-to-loops` + `convert-vector-to-llvm` pipeline 把 linalg.softmax 转
//     `<VL x float>` SIMD 标量循环 → LLVM 自动向量化
//   - 不用手写标量循环：避免之前 MatMulOpLowering 标量循环问题
//
// 公式：
//   y[i,j] = exp(x[i,j] - rowmax[i]) / rowsum[i]
//   rowmax[i] = max_j x[i,j]
//   rowsum[i] = sum_j exp(x[i,j] - rowmax[i])
struct SoftmaxOpLowering : public mlir::OpRewritePattern<mlir::c3::SoftmaxOp> {
    using OpRewritePattern<mlir::c3::SoftmaxOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::SoftmaxOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value output = op.getOut();
        int64_t axis = op.getAxis();

        // linalg.softmax expects IntegerAttr for dimension
        auto dimAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(axis));

        // C3 softmax 是 out-as-operand（无 SSA result），linalg.softmax 是 SSA result 语义
        // 但因为 linalg.softmax 的 outs operand **就是**输出 buffer，
        // 创建 linalg.softmax 后会"填充"output buffer —— resultType 给空数组避免生成新 buffer
        rewriter.create<mlir::linalg::SoftmaxOp>(
            loc,
            /*resultType=*/mlir::TypeRange{},  // 不创建新 result（output 已是 SSA 占位）
            input,
            output,
            dimAttr
        );

        // 抹除原 c3.softmax op（linalg.softmax 已接管）
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// [P0.2 2026-08-30 苏璃珞] CrossEntropyOpLowering：c3.cross_entropy → 手写 fused loop
//
// 公式：out[0] = -1/M * sum_i sum_c target_ic * log(softmax(logits)_ic)
//
// 数值稳定版（max-subtraction）：
//   对每行 i：
//     max_i = max_j logits[i, j]
//     sum_exp_i = sum_j exp(logits[i, j] - max_i)
//     loss_i = -sum_j target[i, j] * log(exp(logits[i, j] - max_i) / sum_exp_i + eps)
//   out[0] = sum_i loss_i / M
//
// 选 fused loop 而非 linalg.softmax + linalg.elementwise + linalg.reduce 是因为：
//   - 一次 row 扫描即可同时算 max/sum_exp/loss（3 个不同算子合并为 1 个 3 级 nested loop）
//   - 避免中间分配 exp(logits) 临时 buffer
//   - 现有 LinalgOneShotGen 缺多输入（logits+target）elementwise + softmax fused 的 op
//     支持，需要新加 linalg generic 比较麻烦
//   - 当前 C3 已有手写 fused 循环的先例（MatMulOpLowering，cblas sgemm 内联等）
struct CrossEntropyOpLowering : public mlir::OpRewritePattern<mlir::c3::CrossEntropyOp> {
    using OpRewritePattern<mlir::c3::CrossEntropyOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::CrossEntropyOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value logits = op.getLogits();
        mlir::Value target = op.getTarget();
        mlir::Value out = op.getOut();
        int64_t M = op.getM();
        int64_t N = op.getN();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i64_type = rewriter.getI64Type();

        mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
        mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
        mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

        // 1) 清零 out[0] = 0
        mlir::Value zero_idx = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
        mlir::Value out_ptr0 = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out,
            mlir::ValueRange{zero_idx});
        mlir::Value fzero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
        rewriter.create<mlir::LLVM::StoreOp>(loc, fzero, out_ptr0);

        // 2) 外层 for i = 0..M, carry 累加 loss
        auto init_loss = fzero;
        auto outer_loop = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1,
            mlir::ValueRange{init_loss});
        rewriter.setInsertionPointToStart(outer_loop.getBody());
        mlir::Value i_idx = outer_loop.getInductionVar();
        mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);
        mlir::Value loss_carry = outer_loop.getRegionIterArgs()[0];

        // 3) 中层 for j = 0..N, 计算 max_i = max_j logits[i, j]（carry = max）
        //    初值用一个非常小的负数（-1e30 等价 -INFINITY）
        mlir::Value neg_inf = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32,
            llvm::APFloat(-1.0e30f));
        auto max_loop = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1,
            mlir::ValueRange{neg_inf});
        rewriter.setInsertionPointToStart(max_loop.getBody());
        mlir::Value j1_idx = max_loop.getInductionVar();
        mlir::Value j1_i64 = indexToI64(rewriter, loc, j1_idx);
        mlir::Value max_carry = max_loop.getRegionIterArgs()[0];
        mlir::Value row_off1 = rewriter.create<mlir::arith::MulIOp>(loc, i_i64,
            rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
        mlir::Value idx1 = rewriter.create<mlir::arith::AddIOp>(loc, row_off1, j1_i64);
        mlir::Value p1 = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, logits,
            mlir::ValueRange{idx1});
        mlir::Value v1 = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, p1);
        // max(a, b) 用 arith.maximumf（与 0 比较 NaN-safe）
        mlir::Value new_max = rewriter.create<mlir::arith::MaximumFOp>(loc, max_carry, v1);
        rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_max});

        rewriter.setInsertionPointAfter(max_loop);
        mlir::Value row_max = max_loop.getResult(0);

        // 4) 中层 for j = 0..N, 计算 sum_exp_i = sum_j exp(logits[i, j] - max_i)
        auto sumexp_loop = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1,
            mlir::ValueRange{fzero});
        rewriter.setInsertionPointToStart(sumexp_loop.getBody());
        mlir::Value j2_idx = sumexp_loop.getInductionVar();
        mlir::Value j2_i64 = indexToI64(rewriter, loc, j2_idx);
        mlir::Value sum_carry = sumexp_loop.getRegionIterArgs()[0];
        mlir::Value row_off2 = rewriter.create<mlir::arith::MulIOp>(loc, i_i64,
            rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
        mlir::Value idx2 = rewriter.create<mlir::arith::AddIOp>(loc, row_off2, j2_i64);
        mlir::Value p2 = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, logits,
            mlir::ValueRange{idx2});
        mlir::Value v2 = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, p2);
        mlir::Value v2_shift = rewriter.create<mlir::arith::SubFOp>(loc, v2, row_max);
        mlir::Value ev2 = rewriter.create<mlir::math::ExpOp>(loc, v2_shift);
        mlir::Value new_sum = rewriter.create<mlir::arith::AddFOp>(loc, sum_carry, ev2);
        rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_sum});

        rewriter.setInsertionPointAfter(sumexp_loop);
        mlir::Value sum_exp = sumexp_loop.getResult(0);
        // inv_sum = 1 / sum_exp（用 reciprocalf；sum_exp > 0 因为 exp 至少有一个 e^0 = 1）
        mlir::Value fone = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));
        mlir::Value inv_sum = rewriter.create<mlir::arith::DivFOp>(loc, fone, sum_exp);

        // 5) 中层 for j = 0..N, 计算 loss_i = -sum_j target[i, j] * log(exp(logits[i, j] - max_i) * inv_sum + eps)
        mlir::Value eps = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32,
            llvm::APFloat(1.0e-7f));
        auto loss_loop = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1,
            mlir::ValueRange{loss_carry});
        rewriter.setInsertionPointToStart(loss_loop.getBody());
        mlir::Value j3_idx = loss_loop.getInductionVar();
        mlir::Value j3_i64 = indexToI64(rewriter, loc, j3_idx);
        mlir::Value loss_carry_inner = loss_loop.getRegionIterArgs()[0];
        mlir::Value row_off3 = rewriter.create<mlir::arith::MulIOp>(loc, i_i64,
            rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
        // logits[i, j] 再次加载
        mlir::Value idx3 = rewriter.create<mlir::arith::AddIOp>(loc, row_off3, j3_i64);
        mlir::Value p3 = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, logits,
            mlir::ValueRange{idx3});
        mlir::Value v3 = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, p3);
        mlir::Value v3_shift = rewriter.create<mlir::arith::SubFOp>(loc, v3, row_max);
        mlir::Value ev3 = rewriter.create<mlir::math::ExpOp>(loc, v3_shift);
        mlir::Value p_ij = rewriter.create<mlir::arith::MulFOp>(loc, ev3, inv_sum);
        mlir::Value p_ij_safe = rewriter.create<mlir::arith::MaximumFOp>(loc, p_ij, eps);
        mlir::Value log_p = rewriter.create<mlir::math::LogOp>(loc, p_ij_safe);
        // target[i, j]
        mlir::Value pt = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, target,
            mlir::ValueRange{idx3});
        mlir::Value t_ij = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, pt);
        mlir::Value product = rewriter.create<mlir::arith::MulFOp>(loc, t_ij, log_p);
        mlir::Value new_loss = rewriter.create<mlir::arith::SubFOp>(loc, loss_carry_inner, product);
        rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_loss});

        rewriter.setInsertionPointAfter(loss_loop);
        mlir::Value new_loss_carry = loss_loop.getResult(0);
        rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_loss_carry});

        rewriter.setInsertionPointAfter(outer_loop);
        // 6) loss_total = outer_loop.result, out[0] = loss_total / M
        mlir::Value loss_total = outer_loop.getResult(0);
        mlir::Value M_f = rewriter.create<mlir::arith::SIToFPOp>(loc, f32,
            rewriter.create<mlir::arith::ConstantIntOp>(loc, M, 64));
        mlir::Value mean_loss = rewriter.create<mlir::arith::DivFOp>(loc, loss_total, M_f);
        rewriter.create<mlir::LLVM::StoreOp>(loc, mean_loss, out_ptr0);

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

static void runC3Lowering(mlir::ModuleOp module) {
    mlir::RewritePatternSet patterns(module.getContext());
    patterns.add<TransposeOpLowering, SumReduceOpLowering, MatMulOpLowering,
                 AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
                 NegOpLowering, ReLUOpLowering, SigmoidOpLowering, TanhOpLowering,
                 ExpOpLowering, LogOpLowering,
                 SoftmaxOpLowering, CrossEntropyOpLowering>(module.getContext());  // [P0.2] 加 Softmax + CrossEntropy lowering
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
        throw std::runtime_error("C3DialectLowering: C3ToLLVM lowering pass failed");
    }
}

static void runPass(mlir::ModuleOp module, std::unique_ptr<mlir::Pass> pass, const char* name) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(std::move(pass));
    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error(std::string("C3DialectLowering: ") + name + " failed");
    }
}

void applyLoweringPipeline(mlir::ModuleOp module, int opt_level) {
    runPass(module, mlir::createStripDebugInfoPass(), "StripDebugInfo");
    runPass(module, mlir::createCanonicalizerPass(), "Canonicalizer");
    runC3Combine(module);  // 1. 运行 JIT 3.0 高层图优化 (DRR)
    runC3Lowering(module); // 2. 运行 JIT 3.0 高层算子到 LLVM 标量/向量循环 of Lowering Pass
    runPass(module, mlir::createCSEPass(), "CSE");
    runPass(module, mlir::createSymbolDCEPass(), "SymbolDCE");
    runPass(module, mlir::createLoopInvariantCodeMotionPass(), "LICM");
    runPass(module, mlir::createSCFForLoopCanonicalizationPass(), "SCFForLoopCanonicalization");
    
    // [Extreme JIT - opt_level >= 4] 能上的优化 Pass 尽可能上满，释放硬件级极致性能
    if (opt_level >= 4) {
        runPass(module, mlir::createControlFlowSinkPass(), "ControlFlowSink");
        runPass(module, mlir::createRemoveDeadValuesPass(), "RemoveDeadValues");
    }

    // [优化 2026-08-16] 移除 ParallelLoopFusionPass。因为 C3DialectLowering 仅生成顺序 scf.for 循环，
    // 无 scf.parallel 循环，此 pass 为 100% no-op，移除它以减少编译期 pass 遍历开销。
    runPass(module, mlir::createSCFToControlFlowPass(), "SCFToCF");

    runPass(module, mlir::createConvertMathToLLVMPass(), "MathToLLVM");
    runPass(module, mlir::createArithToLLVMConversionPass(), "ArithToLLVM");

    runPass(module, mlir::createConvertControlFlowToLLVMPass(), "CFToLLVM");
    runPass(module, mlir::createConvertFuncToLLVMPass(), "FuncToLLVM");
    runPass(module, mlir::createFinalizeMemRefToLLVMConversionPass(), "MemRefToLLVM");

    runPass(module, mlir::createReconcileUnrealizedCastsPass(), "ReconcileUnrealizedCasts");

    // LLVM 转换收尾后再次运行 Canonicalizer & CSE 清理无效转换、类型强转与死代码，精简 IR
    runPass(module, mlir::createCanonicalizerPass(), "CanonicalizerPost");
    runPass(module, mlir::createCSEPass(), "CSEPost");
}

} // namespace c3
} // namespace ct
