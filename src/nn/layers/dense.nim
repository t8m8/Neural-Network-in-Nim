import "../nnenv"
import "../../linalg/matrix"
import layer

type
  Dense* = ref object of Links

proc newDense*(inputDim, outputDim: int; eps: NNFloat = 1e-4): Dense =
  new(result)
  result.inputDim = inputDim
  result.outputDim = outputDim
  result.weights = newMatRandom[NNFloat](inputDim, outputDim, -eps, eps)

proc newDense*(inputDim, outputDim: int; weights: Matrix[NNFloat]): Dense =
  new(result)
  result.inputDim = inputDim
  result.outputDim = outputDim
  result.weights = weights

method compute*(self: Dense, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  incoming * self.weights

method delta*(self: Dense, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} = 
  above * self.weights.t