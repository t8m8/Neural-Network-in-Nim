import "../nnenv"
import "../../utils/mysequtils"

type
  TrResults* = ref object of RootObj
    totalCount*: int
    currentCount*: int
    totalLoss*: NNFloat
    hitCount*: int
    missCount*: int

  Measure* = ref object of RootObj

method `$`*(self: Measure): string {.base.} =
  assert false

method compute*(self: Measure, res: TrResults): NNFloat {.base.} =
  assert false



type
  Formatter* = ref object of RootObj
    measures*: seq[Measure]
    progressWidth: int
    flag: bool

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
    if i.NNFloat <= progress: bar &= "#"
    else: bar &= " "

  let per = res.currentCount * 100 / res.totalCount
  echo $res.currentCount & " / " & $res.totalCount & " (" & $per & "%)\n" &
    "[" & bar & "]" 
  for m in self.measures:
    echo $m & " = " & $m.compute(res)

method epochStart*(self: Formatter, currentEpoch, totalEpochs: int) {.base.} = 
  if not self.flag: return
  echo "Training epoch " & $currentEpoch & " / " & $totalEpochs

method epochEnd*(self: Formatter, currentEpoch, totalEpochs: int) {.base.} =
  if not self.flag: return
  echo ""
