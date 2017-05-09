import "../nnenv"
import "../../linalg/matrix"
import lossfunc
import math, future

type
  BinaryCrossEntropy = ref object of LossFunc

proc newBinaryCrossEntropy*(): BinaryCrossEntropy {.noSideEffect.} =
  new(result)

method loss*(self: BinaryCrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  result = output.reduceRows(0.0, proc(acc, val: NNFloat; row, col: int): NNFloat =
    if expected[row, col] < 1e-5:
      -(1.0 - val).ln
    else:
      -val.ln
  )

method delta*(self: BinaryCrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  result = output.transform((val: NNFloat, row, col) => val - expected[row, col])

method predictFromProbs*(self: BinaryCrossEntropy, probs: Matrix[NNFloat]): Matrix[int] {.noSideEffect.} =
  result = probs.transform(proc(val: NNFloat): int =
    if val >= 0.5: 1
    else: 0
  )