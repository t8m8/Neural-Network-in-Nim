import "../nnenv"
import optimizer
import "../../linalg/matrix"

type
  SGD* = ref object of Optimizer

proc newSGD*(lr: NNFloat = 0.01): SGD {.noSideEffect.} =
  new(result)
  result.lr = lr

method update*(self: var SGD, weights: var Matrix[NNFloat], gradients: var Matrix[NNFloat]) =
  weights = weights - (gradients * self.lr)