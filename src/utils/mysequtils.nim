include sequtils

proc last*[T](self: seq[T]): T {.inline, noSideEffect.} = self[self.len - 1]