include sequtils

proc head*[T](self: seq[T]): T {.inline, noSideEffect.} = self[0]

proc last*[T](self: seq[T]): T {.inline, noSideEffect.} = self[self.len - 1]