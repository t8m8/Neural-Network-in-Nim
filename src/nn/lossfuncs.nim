import ../linalg/matrix

import math, future

type
  LossFunc* = ref object of RootObj

  CrossEntropy = ref object of LossFunc
  BinaryCrossEntropy = ref object of LossFunc

method loss*(self: LossFunc, output: Matrix[float64],
    expected: Matrix[float64]): Matrix[float64] {.base.} =
  assert false

method backward*(self: LossFunc, output: Matrix[float64],
    expected: Matrix[float64]): Matrix[float64] {.base.} =
  assert false

method predictFromProbs*(self: LossFunc, probs: Matrix[float64]):
    Matrix[int] {.base.} =
  assert false

# ==============================================================================

proc newCrossEntropy*(): CrossEntropy {.noSideEffect.} =
  new(result)

method loss*(self: CrossEntropy, output: Matrix[float64],
    expected: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  result = output.reduceRows(0.0, proc(acc, val: float64, row, col: int):
      float64 =
    acc - expected[row, col]*ln(val)
  )

method backward*(self: CrossEntropy, output: Matrix[float64],
    expected: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  result = output.transform((val: float64, row, col) => val - expected[row, col])

method predictFromProbs*(self: CrossEntropy, probs: Matrix[float64]):
    Matrix[int] {.noSideEffect.} =
  result = probs.reduceRows(
    (0.0, 0),
    proc(acc: tuple[max: float64, maxIdx: int], val: float64, row, col: int):
        (float64, int) =
      if val > acc.max:
        result = (val, col)
      else:
        result = acc
  ).transform(
    (val: tuple[max: float64, maxIdx: int]) => val.maxIdx
  )

# ==============================================================================

proc newBinaryCrossEntropy*(): BinaryCrossEntropy {.noSideEffect.} =
  new(result)

method loss*(self: BinaryCrossEntropy, output: Matrix[float64],
    expected: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  result = output.reduceRows(0.0, proc(acc, val: float64, row, col: int):
      float64 =
    if expected[row, col] < 1e-5:
      -(1.0 - val).ln
    else:
      -val.ln
  )

method backward*(self: BinaryCrossEntropy, output: Matrix[float64],
    expected: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  result = output.transform(
      (val: float64, row, col) => val - expected[row, col])

method predictFromProbs*(self: BinaryCrossEntropy, probs: Matrix[float64]):
    Matrix[int] {.noSideEffect.} =
  result = probs.transform(proc(val: float64): int =
    if val >= 0.5: 1
    else: 0
  )