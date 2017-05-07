import nnenv
import "measures/measures"

type
  Options* = ref object of RootObj
    epochs*: int
    batchSize*: int
    formatter*: Formatter

proc newOptions*(epochs, batchSize: int): Options =
  new(result)
  result.epochs = epochs
  result.batchSize = batchSize
  result.formatter = newFormatter()

proc addMeasure*(self: var Options, m: Measure) =
  self.formatter.addMeasure(m)