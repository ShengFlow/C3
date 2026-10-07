//===- C3Combine.td - Pattern Match Optimizations for C3 ----*- tablegen -*-===//
//
// Defines pattern match optimizations for the C3 dialect using Declarative
// Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef C3_COMBINE
#define C3_COMBINE

include "mlir/IR/PatternBase.td"
include "C3/C3Ops.td"

//===----------------------------------------------------------------------===//
// C3 Pattern-Match and Rewrite Rules
//===----------------------------------------------------------------------===//

// 规则 1: 双重转置消去 (Tensor SSA 语义)
// transpose_tensor(transpose_tensor(x, dest1, M1, N1, d0, d1), dest2, M2, N2, d2, d3) -> x
def DoubleTransposeTensorOptPattern : Pat<
  (C3_TransposeTensorOp (C3_TransposeTensorOp $arg, $dest1, $M1, $N1, $d0, $d1), $dest2, $M2, $N2, $d2, $d3),
  (replaceWithValue $arg)
>;

// 规则 2: 双重取负消去 (Tensor SSA 语义)
// neg_tensor(neg_tensor(x, dest1), dest2) -> x
def DoubleNegTensorOptPattern : Pat<
  (C3_NegTensorOp (C3_NegTensorOp $arg, $dest1), $dest2),
  (replaceWithValue $arg)
>;

// 规则 3: ReLU 幂等折叠 (Tensor SSA 语义)
// relu_tensor(relu_tensor(x, dest1), dest2) -> relu_tensor(x, dest2)
def ReluIdempotentTensorOptPattern : Pat<
  (C3_ReLUTensorOp (C3_ReLUTensorOp $arg, $dest1), $dest2),
  (C3_ReLUTensorOp $arg, $dest2)
>;

#endif // C3_COMBINE
