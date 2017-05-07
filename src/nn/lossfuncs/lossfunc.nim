import "../nnenv"
import "../../linalg/matrix"

type
  LossFunc* = ref object of RootObj

method loss*(self: LossFunc, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

method delta*(self: LossFunc, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

method predictFromProbs*(self: LossFunc, probs: Matrix[NNFloat]): Matrix[int] {.base.} =
  assert false

