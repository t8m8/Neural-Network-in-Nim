import "../nnenv"
import optimizer
import "../../linalg/matrix"

import math

type
  Adam* = ref object of Optimizer
    beta1*: NNFloat
    beta2*: NNFloat
    iter*: int
    v*: Matrix[NNFloat]
    m*: Matrix[NNFloat]

proc newAdam*(lr: NNFloat = 0.001, beta1: NNFloat = 0.9, beta2: NNFloat = 0.999): Adam {.noSideEffect.} =
  new(result)
  result.lr = lr
  result.beta1 = beta1
  result.beta2 = beta2
  result.iter = 0

method update*(self: var Adam, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  if self.v.isNil:
    self.m = newMat[NNFloat](gradients.row, gradients.col)
    self.v = newMat[NNFloat](gradients.row, gradients.col)

  self.iter.inc
  let lr = self.lr * (1.0 - pow(self.beta2, self.iter.NNFloat)).sqrt / (1.0 - pow(self.beta1, self.iter.NNFloat))

  self.m = self.m + (1.0 - self.beta1) @* (gradients - self.m)
  self.v = self.v + (1.0 - self.beta2) @* (gradients ^ 2 - self.v)
  gradients = gradients - lr*self.m @/ (self.v.map(sqrt) + 1e-7)
