import nnenv
import "../linalg/matrix"

import math, sequtils, future

type
  Layer* = ref object of RootObj

  Activation* = ref object of Layer

  Identity = ref object of Activation
  Sigmoid = ref object of Activation
  ReLU = ref object of Activation
  Softmax = ref object of Activation

  Links* = ref object of Layer
    inputDim*: int
    outputDim*: int
    weights*: Matrix[NNFloat]

  Dense* = ref object of Links

proc `[]=`*(self: var Links, row, col: int, w: NNFloat) =
  self.weights[row, col] = w

method forward*(self: Layer, incoming: Matrix[NNFloat]):
    Matrix[NNFloat] {.base.} =
  assert false

method backward*(self: Layer, outgoing: Matrix[NNFloat],
    above: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

# ==============================================================================

proc newIdentity*(): Identity {.noSideEffect.} =
  new(result)

method forward*(self: Identity, incoming: Matrix[NNFloat]):
    Matrix[NNFloat] {.noSideEffect.} =
  incoming

method backward*(self: Identity, outgoing: Matrix[NNFloat],
    above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  above

# ==============================================================================

proc newSigmoid*(): Sigmoid {.noSideEffect.} =
  new(result)

method forward*(self: Sigmoid, incoming: Matrix[NNFloat]):
    Matrix[NNFloat] {.noSideEffect.} =
  incoming.transform((val: NNFloat) => 1.0 / (1.0 + exp(-val)))

method backward*(self: Sigmoid, outgoing: Matrix[NNFloat],
    above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  outgoing.transform((val: NNFloat, row, col: int) =>
    val * (1.0 - val) * above[row, col]
  )

# ==============================================================================

proc newReLU*(): ReLU {.noSideEffect.} =
  new(result)

method forward*(self: ReLU, incoming: Matrix[NNFloat]):
    Matrix[NNFloat] {.noSideEffect.} =
  incoming.transform((val: NNFloat) => max(0, val))

method backward*(self: ReLU, outgoing: Matrix[NNFloat],
    above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  outgoing.transform((val: NNFloat) => ord(val > 0).NNFloat)

# ==============================================================================

proc newSoftmax*(): Softmax {.noSideEffect.} =
  new(result)

method forward*(self: Softmax, incoming: Matrix[NNFloat]):
    Matrix[NNFloat] {.noSideEffect.} =
  let
    maxes = incoming.reduceRows(0.0, (acc, val) => max(acc, val))
    exps = incoming.transform((val, row, _) => exp(val - maxes[row, 0]))
    sums = exps.reduceRows(0.0, (acc, val) => acc + val)
  result = exps.transform((val, row, _) => val / sums[row, 0])

method backward*(self: Softmax, outgoing: Matrix[NNFloat],
    above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  let
    delta = outgoing * above
    sums = delta.reduceRows(0.0, (acc, val) => acc + val)
  result = delta.transform(
      (val, row, col) => val - outgoing[row, col] * sums[row, 0])

# ==============================================================================

proc newDense*(inputDim, outputDim: int, eps: NNFloat = 1e-4): Dense =
  new(result)
  let d = 1.0 / inputDim.NNFloat
  result.inputDim = inputDim
  result.outputDim = outputDim
  result.weights = newMatRandom[NNFloat](inputDim, outputDim, -d, d)

proc newDense*(inputDim, outputDim: int, weights: Matrix[NNFloat]): Dense =
  new(result)
  result.inputDim = inputDim
  result.outputDim = outputDim
  result.weights = weights

method forward*(self: Dense, incoming: Matrix[NNFloat]):
    Matrix[NNFloat] {.noSideEffect.} =
  incoming * self.weights

method backward*(self: Dense, outgoing: Matrix[NNFloat],
    above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  above * self.weights.t
