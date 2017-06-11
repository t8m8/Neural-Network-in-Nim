import "../linalg/matrix"

import math, future

type
  Optimizer* = ref object of RootObj
    lr*: float64 # learning rate

  SGD* = ref object of Optimizer

  Momentum* = ref object of Optimizer
    v*: Matrix[float64]
    momentum*: float64

  AdaGrad* = ref object of Optimizer
    h*: Matrix[float64]

  Adam* = ref object of Optimizer
    beta1*: float64
    beta2*: float64
    iter*: int
    v*: Matrix[float64]
    m*: Matrix[float64]

method update*(self: var Optimizer, weights: var Matrix[float64],
    gradients: var Matrix[float64]) {.base.} =
  assert false

# ==============================================================================

proc newSGD*(lr: float64 = 0.01): SGD {.noSideEffect.} =
  new(result)
  result.lr = lr

method update*(self: var SGD, weights: var Matrix[float64],
    gradients: var Matrix[float64]) =
  weights = weights - (gradients * self.lr)

# ==============================================================================

proc newMomentum*(lr: float64 = 0.01, momentum: float64 = 0.9):
    Momentum {.noSideEffect.} =
  new(result)
  result.lr = lr
  result.momentum = momentum

method update*(self: var Momentum, weights: var Matrix[float64],
    gradients: var Matrix[float64]) =
  if self.v.isNil:
    self.v = newMat[float64](gradients.row, gradients.col)

  self.v = self.momentum * self.v - self.lr * gradients
  weights = self.v

# ==============================================================================

proc newAdagrad*(lr: float64 = 0.01): AdaGrad {.noSideEffect.} =
  new(result)
  result.lr = lr

method update*(self: var AdaGrad, weights: var Matrix[float64],
    gradients: var Matrix[float64]) =
  if self.h.isNil:
    self.h = newMat[float64](gradients.row, gradients.col)

  self.h = self.h + gradients @* gradients
  weights = weights - (self.lr * gradients @/
      (self.h.map((a) => sqrt(a)) + 1e-7))

# ==============================================================================

proc newAdam*(lr: float64 = 0.001, beta1: float64 = 0.9, beta2: float64 = 0.999):
    Adam {.noSideEffect.} =
  new(result)
  result.lr = lr
  result.beta1 = beta1
  result.beta2 = beta2
  result.iter = 0

method update*(self: var Adam, weights: var Matrix[float64],
    gradients: var Matrix[float64]) =
  if self.v.isNil:
    self.m = newMat[float64](gradients.row, gradients.col)
    self.v = newMat[float64](gradients.row, gradients.col)

  self.iter.inc
  let lr = self.lr * (1.0 - pow(self.beta2, self.iter.float64)).sqrt /
      (1.0 - pow(self.beta1, self.iter.float64))

  self.m = self.m + (1.0 - self.beta1) @* (gradients - self.m)
  self.v = self.v + (1.0 - self.beta2) @* (gradients ^ 2 - self.v)
  gradients = gradients - lr*self.m @/ (self.v.map((a) => sqrt(a)) + 1e-7)

