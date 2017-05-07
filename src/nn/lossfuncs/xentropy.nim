import "../nnenv"
import "../../linalg/matrix"
import lossfunc
import math

type
  CrossEntropy = ref object of LossFunc

proc newCrossEntropy*(): CrossEntropy {.noSideEffect.} =
  new(result)

method loss*(self: CrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  result = output.reduceRows(0.0, proc(acc, val: NNFloat; row, col: int): NNFloat =
    if expected[row, col] > 0.0:
      acc - ln(val)
    else:
      acc
  )

method delta*(self: CrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  result = output.transform(proc(val: NNFloat; row, col: int): NNFloat = val - expected[row, col])

method predictFromProbs*(self: CrossEntropy, probs: Matrix[NNFloat]): Matrix[int] {.noSideEffect.} =
  result = probs.reduceRows(
    (0.0, 0),
    proc(acc: tuple[max: NNFloat, maxIdx: int]; val: NNFloat; row, col: int): tuple[max: NNFloat, maxIdx: int] =
      if val > acc.max:
        result = (val, col)
      else:
        result = acc
  ).transform(
    proc(val: tuple[max: NNFloat, maxIdx: int]): int =
      val.maxIdx 
  )