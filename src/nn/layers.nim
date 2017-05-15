import nnenv
import "../linalg/matrix"

import math, sequtils, future

type
  Layer* = ref object of RootObj

  Activation* = ref object of Layer

  Identity = ref object of Activation
  Sigmoid = ref object of Activation
  ReLU = ref object of Activation

  Links* = ref object of Layer
    inputDim*: int
    outputDim*: int
    weights*: Matrix[NNFloat]

  Dense* = ref object of Links

proc `[]=`*(self: var Links; row, col: int; w: NNFloat) =
  self.weights[row, col] = w

method forward*(self: Layer, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false

method backward*(self: Layer, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.base.} =
  assert false


proc newIdentity*(): Identity {.noSideEffect.} =
  new(result)

method forward*(self: Identity, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  incoming

method backward*(self: Identity, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  above

proc newSigmoid*(): Sigmoid {.noSideEffect.} =
  new(result)

method forward*(self: Sigmoid, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  incoming.transform((val: NNFloat) => 1.0 / (1.0 + exp(-val)))

method backward*(self: Sigmoid, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  outgoing.transform((val: NNFloat, row, col: int) =>
    val * (1.0 - val) * above[row, col]
  )

proc newReLU*(): ReLU {.noSideEffect.} =
  new(result)

method forward*(self: ReLU, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  incoming.transform((val: NNFloat) => max(0, val))

method backward*(self: ReLU, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  outgoing.transform((val: NNFloat) => ord(val > 0).NNFloat)

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

method forward*(self: Dense, incoming: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  incoming * self.weights

method backward*(self: Dense, outgoing: Matrix[NNFloat], above: Matrix[NNFloat]): Matrix[NNFloat] {.noSideEffect.} =
  above * self.weights.t

