import "../nnenv"
import optimizer
import "../../linalg/matrix"

import math

type
  AdaGrad* = ref object of Optimizer
    h*: Matrix[NNFloat]

proc newAdagrad*(lr: NNFloat = 0.01): AdaGrad {.noSideEffect.} =
  new(result)
  result.lr = lr

method update*(self: var AdaGrad, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  if self.h.isNil:
    self.h = newMat[NNFloat](gradients.row, gradients.col)
  
  self.h = self.h + gradients @* gradients
  weights = weights - (self.lr * gradients @/ (self.h.map(sqrt) + 1e-7))
