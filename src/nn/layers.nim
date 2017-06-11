import ../linalg/matrix

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
    weights*: Matrix[float64]

  Dense* = ref object of Links

proc `[]=`*(self: var Links, row, col: int, w: float64) =
  self.weights[row, col] = w

method forward*(self: Layer, incoming: Matrix[float64]):
    Matrix[float64] {.base.} =
  assert false

method backward*(self: Layer, outgoing: Matrix[float64],
    above: Matrix[float64]): Matrix[float64] {.base.} =
  assert false

# ==============================================================================

proc newIdentity*(): Identity {.noSideEffect.} =
  new(result)

method forward*(self: Identity, incoming: Matrix[float64]):
    Matrix[float64] {.noSideEffect.} =
  incoming

method backward*(self: Identity, outgoing: Matrix[float64],
    above: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  above

# ==============================================================================

proc newSigmoid*(): Sigmoid {.noSideEffect.} =
  new(result)

method forward*(self: Sigmoid, incoming: Matrix[float64]):
    Matrix[float64] {.noSideEffect.} =
  incoming.transform((val: float64) => 1.0 / (1.0 + exp(-val)))

method backward*(self: Sigmoid, outgoing: Matrix[float64],
    above: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  outgoing.transform((val: float64, row, col: int) =>
    val * (1.0 - val) * above[row, col]
  )

# ==============================================================================

proc newReLU*(): ReLU {.noSideEffect.} =
  new(result)

method forward*(self: ReLU, incoming: Matrix[float64]):
    Matrix[float64] {.noSideEffect.} =
  incoming.transform((val: float64) => max(0, val))

method backward*(self: ReLU, outgoing: Matrix[float64],
    above: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  outgoing.transform(proc(val: float64, row, col: int): float64 =
    if val > 0: above[row, col]
    else: 0
  )

# ==============================================================================

proc newSoftmax*(): Softmax {.noSideEffect.} =
  new(result)

method forward*(self: Softmax, incoming: Matrix[float64]):
    Matrix[float64] {.noSideEffect.} =
  let
    maxes = incoming.reduceRows(0.0, (acc, val) => max(acc, val))
    exps = incoming.transform((val, row, _) => exp(val - maxes[row, 0]))
    sums = exps.reduceRows(0.0, (acc, val) => acc + val)
  result = exps.transform((val, row, _) => val / sums[row, 0])

method backward*(self: Softmax, outgoing: Matrix[float64],
    above: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  let
    delta = outgoing * above
    sums = delta.reduceRows(0.0, (acc, val) => acc + val)
  result = delta.transform(
      (val, row, col) => val - outgoing[row, col] * sums[row, 0])

# ==============================================================================

proc newDense*(inputDim, outputDim: int): Dense =
  new(result)
  let d = 1.0 / inputDim.float64
  result.inputDim = inputDim
  result.outputDim = outputDim
  result.weights = newMatRandom[float64](inputDim, outputDim, -d, d)

proc newDense*(inputDim, outputDim: int, weights: Matrix[float64]): Dense =
  new(result)
  result.inputDim = inputDim
  result.outputDim = outputDim
  result.weights = weights

method forward*(self: Dense, incoming: Matrix[float64]):
    Matrix[float64] {.noSideEffect.} =
  incoming * self.weights

method backward*(self: Dense, outgoing: Matrix[float64],
    above: Matrix[float64]): Matrix[float64] {.noSideEffect.} =
  above * self.weights.t
