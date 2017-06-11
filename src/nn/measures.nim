import "../utils/mysequtils"

import strutils, terminal

type
  TrResults* = ref object of RootObj
    totalCount*: int
    currentCount*: int
    totalLoss*: float64
    hitCount*: int
    missCount*: int

  Formatter* = ref object of RootObj
    measures*: seq[Measure]
    progressWidth: int
    flag: bool

  Measure* = ref object of RootObj

  Accuracy* = ref object of Measure

  MeanLoss* = ref object of Measure

method `$`*(self: Measure): string {.base.} =
  assert false

method compute*(self: Measure, res: TrResults): float64 {.base.} =
  assert false

# ==============================================================================

proc newFormatter*(): Formatter =
  new(result)
  result.measures = @[]
  result.progressWidth = 80
  result.flag = true

proc addMeasure*(self: var Formatter, m: Measure) =
  self.measures.add(m)

proc on*(self: Formatter) =
  self.flag = true

proc off*(self: Formatter) =
  self.flag = false

method batchEnd*(self: Formatter, res: TrResults) {.base.} =
  if not self.flag: return
  var
    progress = res.currentCount * self.progressWidth / res.totalCount
    bar = ""
  for i in 0..self.progressWidth:
    if i.float64 <= progress: bar &= "#"
    else: bar &= " "

  let per = res.currentCount * 100 / res.totalCount
  var outstr = $res.currentCount & " / " & $res.totalCount &
      " (" & $per & "%)\n" & "[" & bar & "]\n"
  for m in self.measures:
    outstr &= $m & " = " & $m.compute(res) & "\n"

  stdout.write outstr
  for i in 0..<outstr.countLines:
    stdout.cursorUp()
    stdout.eraseLine()

method epochStart*(self: Formatter, currentEpoch, totalEpochs: int) {.base.} =
  if not self.flag: return
  stdout.write "Training epoch " & $currentEpoch & " / " & $totalEpochs & "\n"

method epochEnd*(self: Formatter, currentEpoch, totalEpochs: int) {.base.} =
  if not self.flag: return
  stdout.cursorUp()
  stdout.eraseLine()

# ==============================================================================

proc newAccuracy*(): Accuracy =
  new(result)

method compute*(self: Accuracy, res: TrResults): float64 {.noSideEffect.} =
  res.hitCount.float64 / res.currentCount.float64

method `$`*(self: Accuracy): string {.noSideEffect.} = "accuracy"

proc newMeanLoss*(): MeanLoss =
  new(result)

method compute*(self: MeanLoss, res: TrResults): float64 {.noSideEffect.} =
  res.totalLoss / res.currentCount.float64

method `$`*(self: MeanLoss): string {.noSideEffect.} = "mean loss"
