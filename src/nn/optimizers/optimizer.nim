import "../nnenv"
import "../../linalg/matrix"

type
  Optimizer* = ref object of RootObj
    lr*: NNFloat # learning rate

method update*(self: var Optimizer, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) {.base.} =
  assert false
