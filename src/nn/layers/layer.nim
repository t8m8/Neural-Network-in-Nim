import "../nnenv"
import "../../linalg/matrix"

type
  Layer* = ref object of RootObj

  Activation* = ref object of Layer

  Links* = ref object of Layer
    inputDim*: int
    outputDim*: int
    weights*: Matrix[NNFloat]
  
method compute*(self: Layer, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

method delta*(self: Layer, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

proc `[]=`*(self: var Links; row, col: int; w: NNFloat) =
  self.weights[row, col] = w