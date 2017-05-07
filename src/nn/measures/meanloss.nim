import "../nnenv"
import measure

type
  MeanLoss* = ref object of Measure

proc newMeanLoss*(): MeanLoss =
  new(result)

method compute*(self: MeanLoss, res: TrResults): NNFloat {.noSideEffect.} =
  res.totalLoss / res.currentCount.NNFloat

method `$`*(self: MeanLoss): string {.noSideEffect.} = "mean loss"
