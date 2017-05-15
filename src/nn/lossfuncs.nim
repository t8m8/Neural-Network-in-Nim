import nnenv
import "../linalg/matrix"

import math, future

type
  LossFunc* = ref object of RootObj

  CrossEntropy = ref object of LossFunc
  BinaryCrossEntropy = ref object of LossFunc

method loss*(self: LossFunc, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

method backward*(self: LossFunc, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

method predictFromProbs*(self: LossFunc, probs: Matrix[NNFloat]): Matrix[int] {.base.} =
  assert false

proc newCrossEntropy*(): CrossEntropy {.noSideEffect.} =
  new(result)

method loss*(self: CrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  result = output.reduceRows(0.0, proc(acc, val: NNFloat, row, col: int): NNFloat =
    if expected[row, col] > 0.0:
      acc - ln(val)
    else:
      acc
  )

method backward*(self: CrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  result = output.transform((val: NNFloat, row, col) => val - expected[row, col])

method predictFromProbs*(self: CrossEntropy, probs: Matrix[NNFloat]): Matrix[int] {.noSideEffect.} =
  result = probs.reduceRows(
    (0.0, 0),
    proc(acc: tuple[max: NNFloat, maxIdx: int], val: NNFloat, row, col: int): tuple[max: NNFloat, maxIdx: int] =
      if val > acc.max:
        result = (val, col)
      else:
        result = acc
  ).transform(
    (val: tuple[max: NNFloat, maxIdx: int]) => val.maxIdx
  )

proc newBinaryCrossEntropy*(): BinaryCrossEntropy {.noSideEffect.} =
  new(result)

method loss*(self: BinaryCrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  result = output.reduceRows(0.0, proc(acc, val: NNFloat, row, col: int): NNFloat =
    if expected[row, col] < 1e-5:
      -(1.0 - val).ln
    else:
      -val.ln
  )

method backward*(self: BinaryCrossEntropy, output: Matrix[NNFloat], expected: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  result = output.transform((val: NNFloat, row, col) => val - expected[row, col])

method predictFromProbs*(self: BinaryCrossEntropy, probs: Matrix[NNFloat]): Matrix[int] {.noSideEffect.} =
  result = probs.transform(proc(val: NNFloat): int =
    if val >= 0.5: 1
    else: 0
  )