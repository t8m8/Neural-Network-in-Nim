import random, typetraits, sequtils, future

type
  Matrix*[T] = ref object of RootObj
    col*: int
    row*: int
    elm: seq[T]

const EPS = 1e-8

proc `$`*[T](self: Matrix[T]): string {.noSideEffect.} =
  "Matrix[" & T.type.name & "]{row: " & $self.row & ", col: " & $self.col & ", elm: " & $self.elm & "}"

proc newMat*[T](row, col: int): Matrix[T] {.noSideEffect.} =
  new(result)
  result.row = row
  result.col = col
  result.elm = newSeq[T](row*col)

proc newMat*[T](row, col: int, elm: seq[T]): Matrix[T] {.noSideEffect.} =
  new(result)
  result.row = row
  result.col = col
  result.elm = elm

proc newMatRandom*[T](row, col: int, min, max: T): Matrix[T] =
  # [min, max)
  new(result)
  result.row = row
  result.col = col
  var elm: seq[T] = @[]
  for i in 0..<row*col:
    elm.add random(max - min) + min
  result.elm = elm

proc at*[T](self: Matrix[T], i, j: int): T {.inline, deprecated.} = self.elm[i*self.col + j]

proc `[]`*[T](self: Matrix[T], i, j: int): T {.inline, noSideEffect.} = self.elm[i*self.col + j]

proc setAt*[T](self: var Matrix[T], i, j: int, val: T) {.inline, deprecated.} =
  self.elm[i*self.col + j] = val

proc `[]=`*[T](self: var Matrix[T], i, j: int, val: T) {.inline.} =
  self.elm[i*self.col + j] = val

proc `[]=`*[T](self: var Matrix[T], i: int, val: seq[T]) {.inline.} =
  for j in 0..<val.len:
    self[i, j] = val[j]

proc `==`*[T](x, y: var Matrix[T]): bool =
  if x.row != y.row or x.col != y.col:
    return false
  else:
    for i in 0..<x.elm.len:
      if x.elm[i] != y.elm[i]:
        return false
  true

proc `~=`*[T](x, y: var Matrix[T]): bool =
  if x.row != y.row or x.col != y.col:
    return false
  else:
    for i in 0..<x.elm.len:
      if abs(x.elm[i] - y.elm[i]) > EPS:
        return false
  true

proc I*[T](dim: int): Matrix[T] {.noSideEffect.} =
  result = newMat[T](dim, dim)
  for i in 0..<dim:
    result[i,i] = 1.0

proc `+`*[T](x, y: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  for i in 0..<x.elm.len:
    result.elm[i] = x.elm[i] + y.elm[i]

proc `+`*[T](x: Matrix[T], y: T): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  result.elm = x.elm.map((val: float64) => val+y)

proc `+`*[T](y: T, x: Matrix[T]): Matrix[T] {.noSideEffect.} = x + y

proc `-`*[T](x, y: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  for i in 0..<x.elm.len:
    result.elm[i] = x.elm[i] - y.elm[i]

proc `-`*[T](x: Matrix[T], y: T): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  result.elm = x.elm.map((val: float64) => val-y)

proc `-`*[T](y: T, x: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  result.elm = x.elm.map((val: float64) => y-val)

proc `*`*[T](x, y: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, y.col)
  for i in 0..<x.row:
    for k in 0..<x.col:
      for j in 0..<y.col:
        result[i,j] = result[i,j] + x[i,k] * y[k,j]

proc `*`*[T](x: Matrix[T], y: T): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  result.elm = x.elm.map((val:float64) => val*y)

proc `*`*[T](y: T, x: Matrix[T]): Matrix[T] {.noSideEffect.} = x * y

proc `^`*[T](x: Matrix[T], n: int): Matrix[T] {.noSideEffect.} =
  # TODO: exponentiation by squaring
  result = I[T](x.row)
  for i in 0..<n:
    result = result * x

proc `/`*[T](x: Matrix[T], y: T): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, x.col)
  result.elm = x.elm.map((val: float64) => val/y)

proc `/`*[T](y: T, x: Matrix[T]): Matrix[T] {.noSideEffect.} = x / y

proc `@*`*[T](x, y: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, y.col)
  for i in 0..<x.elm.len:
    result.elm[i] = x.elm[i] * y.elm[i]

proc `@/`*[T](x, y: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](x.row, y.col)
  for i in 0..<x.elm.len:
    result.elm[i] = x.elm[i] / y.elm[i]

proc map*[T](self: Matrix[T], f: T -> T): Matrix[T] {.noSideEffect.} =
  result = newMat[T](self.row, self.col)
  result.elm = self.elm.map(f)

proc t*[T](self: Matrix[T]): Matrix[T] {.noSideEffect.} =
  result = newMat[T](self.col, self.row)
  for i in 0..<self.row:
    for j in 0..<self.col:
      result[j,i] = self[i,j]

proc reduce*[T, S](self: Matrix[T], init: S, f: (S, T, int, int) -> S): S {.noSideEffect.} =
  result = init
  for i in 0..<self.row:
    for j in 0..<self.col:
      result = f(result, self[i,j], i, j)

proc reduce*[T, S](self: Matrix[T], init: S, f: (S, T) -> S): S {.noSideEffect.} =
  self.reduce(init, (acc: S, val: T, row, col) => f(acc, val))

proc reduceRows*[T, S](self: Matrix[T], init: S, f: (S, T, int, int) -> S): Matrix[S] {.noSideEffect.} =
  result = newMat[S](self.row, 1)
  for i in 0..<self.row:
    result[i,0] = init
    for j in 0..<self.col:
      result[i,0] = f(result[i,0], self[i,j], i, j)

proc reduceRows*[T, S](self: Matrix[T], init: S, f: (S, T) -> S): Matrix[S] {.noSideEffect.} =
  self.reduceRows(init, (acc: S, val: T, row, col) => f(acc, val))

proc transform*[T, S](self: Matrix[T], f: (T, int, int) -> S): Matrix[S] {.noSideEffect.} =
  result = newMat[S](self.row, self.col)
  for i in 0..<self.row:
    for j in 0..<self.col:
      result[i,j] = f(self[i,j], i, j)

proc transform*[T, S](self: Matrix[T], f: T -> S): Matrix[S] {.noSideEffect.} =
  self.transform((val: T, row, col) => f(val))

proc slice*[T](self: Matrix[T], lb, ub: int): Matrix[T] {.noSideEffect.} =
  # [lb, ub]
  result = newMat[T](ub - lb + 1, self.col)
  var pos = 0
  for i in lb..ub:
    for j in 0..<self.col:
      result[pos,j] = self[i,j]
    pos.inc
