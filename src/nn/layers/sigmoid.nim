import "../nnenv"
import "../../linalg/matrix"
import layer
import math, sequtils, future

type
  Sigmoid = ref object of Activation

proc newSigmoid*(): Sigmoid {.noSideEffect.} =
  new(result)

method compute*(self: Sigmoid, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  incoming.transform((val: NNFloat) => 1.0 / (1.0 + exp(-val)))

method delta*(self: Sigmoid, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  outgoing.transform((val: NNFloat, row, col: int) =>
    val * (1.0 - val) * above[row, col]
  )