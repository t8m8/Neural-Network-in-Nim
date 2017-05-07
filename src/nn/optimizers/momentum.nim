import "../nnenv"
import optimizer
import "../../linalg/matrix"

type
  Momentum* = ref object of Optimizer
    v*: Matrix[NNFloat]
    momentum*: NNFloat

proc newMomentum*(lr: NNFloat = 0.01, momentum: NNFloat = 0.9): Momentum {.noSideEffect.} =
  new(result)
  result.lr = lr
  result.momentum = momentum

method update*(self: var Momentum, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  if self.v.isNil:
    self.v = newMat[NNFloat](gradients.row, gradients.col)
  
  self.v = self.momentum * self.v - self.lr * gradients
  weights = self.v
