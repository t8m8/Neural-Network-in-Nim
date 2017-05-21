include sequtils

proc head*[T](self: seq[T]): T {.inline, noSideEffect.} = self[0]

proc last*[T](self: seq[T]): T {.inline, noSideEffect.} = self[self.len - 1]

proc oneHot*[T](k, idx: int): seq[T] {.noSideEffect.} =
  result = newSeq[T](k)
  result[idx] = 1.0