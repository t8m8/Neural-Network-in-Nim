import "../nnenv"
import "../../linalg/matrix"
import layer
import math, sequtils

type
  Identity = ref object of Activation

proc newIdentity*(): Identity {.noSideEffect.} =
  new(result)

method compute*(self: Identity, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  incoming

method delta*(self: Identity, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  above