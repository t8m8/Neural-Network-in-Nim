import "../nnenv"
import "../../linalg/matrix"
import layer
import math, sequtils

type
  ReLU = ref object of Activation

proc newReLU*(): ReLU {.noSideEffect.} =
  new(result)

method compute*(self: ReLU, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  incoming.transform(proc(val: NNFloat): NNFloat = max(0, val))

method delta*(self: ReLU, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  outgoing.transform(proc(val: NNFloat): NNFloat = ord(val > 0).NNFloat)