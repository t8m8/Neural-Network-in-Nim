import nnenv
import "../linalg/matrix"

import math, future

type
  Optimizer* = ref object of RootObj
    lr*: NNFloat # learning rate

  SGD* = ref object of Optimizer

  Momentum* = ref object of Optimizer
    v*: Matrix[NNFloat]
    momentum*: NNFloat

  AdaGrad* = ref object of Optimizer
    h*: Matrix[NNFloat]

  Adam* = ref object of Optimizer
    beta1*: NNFloat
    beta2*: NNFloat
    iter*: int
    v*: Matrix[NNFloat]
    m*: Matrix[NNFloat]

method update*(self: var Optimizer, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) {.base.} =
  assert false

proc newSGD*(lr: NNFloat = 0.01): SGD {.noSideEffect.} =
  new(result)
  result.lr = lr

method update*(self: var SGD, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  weights = weights - (gradients * self.lr)

proc newMomentum*(lr: NNFloat = 0.01, momentum: NNFloat = 0.9): Momentum {.noSideEffect.} =
  new(result)
  result.lr = lr
  result.momentum = momentum

method update*(self: var Momentum, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  if self.v.isNil:
    self.v = newMat[NNFloat](gradients.row, gradients.col)

  self.v = self.momentum * self.v - self.lr * gradients
  weights = self.v

proc newAdagrad*(lr: NNFloat = 0.01): AdaGrad {.noSideEffect.} =
  new(result)
  result.lr = lr

method update*(self: var AdaGrad, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  if self.h.isNil:
    self.h = newMat[NNFloat](gradients.row, gradients.col)

  self.h = self.h + gradients @* gradients
  weights = weights - (self.lr * gradients @/ (self.h.map((a) => sqrt(a)) + 1e-7))

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
  gradients = gradients - lr*self.m @/ (self.v.map((a) => sqrt(a)) + 1e-7)

