import network
import "layers/layers", "lossfuncs/lossfuncs", "optimizers/optimizers"

type
  NNBuilder* = ref object of RootObj
    layers: seq[Layer]

  NNBuilderWithLossFunc* = ref object of NNBuilder
    lossfunc: LossFunc

  NNBuilderWithOptimizer* = ref object of NNBuilderWithLossFunc
    optimizer: Optimizer

proc newNNBuilder*(): NNBuilder =
  new(result)
  result.layers = @[]
  result.layers.add(newIdentity())

proc add*(self: NNBuilder, layer: Layer): NNBuilder =
  new(result)
  result.layers = self.layers
  result.layers.add(layer)

proc minimize*(self: NNBuilder, lossfunc: LossFunc): NNBuilderWithLossFunc =
  new(result)
  result.layers = self.layers
  result.lossfunc = lossfunc

proc optimize*(self: NNBuilderWithLossFunc, optimizer: Optimizer): NNBuilderWithOptimizer =
  new(result)
  result.layers = self.layers
  result.lossfunc = self.lossfunc
  result.optimizer = optimizer

proc build*(self: NNBuilderWithOptimizer): Network =
  newNetwork(self.layers, self.lossfunc, self.optimizer)