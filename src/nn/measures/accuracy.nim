import "../nnenv"
import measure

type
  Accuracy* = ref object of Measure

proc newAccuracy*(): Accuracy =
  new(result)

method compute*(self: Accuracy, res: TrResults): NNFloat {.noSideEffect.} =
  res.hitCount.NNFloat / res.currentCount.NNFloat

method `$`*(self: Accuracy): string {.noSideEffect.} = "accuracy"
