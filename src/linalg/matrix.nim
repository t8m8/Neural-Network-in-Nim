import random, typetraits, sequtils, future

include neo

proc `[]=`*[T](self: var Matrix[T], i: int, val: Vector[T]) {.inline.} =
  for j in 0..<val.len:
    self[i, j] = val[j]

proc oneHot*[T](N, idx: int): Vector[T] =
  makeVector(N, proc(i: int): float64 =
    if i == idx: 1.0
    else: 0.0)

proc `+`*[T](x: Matrix[T], y: T): Matrix[T] {.noSideEffect.} =
  matrix[T](x.order, x.M, x.N, newSeq[T](x.N * x.M)).map((v: float64) => v+y)

proc `+`*[T](y: T, x: Matrix[T]): Matrix[T] {.noSideEffect.} = x + y

proc `^`*[T](x: Matrix[T], n: int): Matrix[T] {.noSideEffect.} =
  # TODO: exponentiation by squaring
  assert(x.M == x.N)
  result = eye[T](x.M)
  for i in 0..<n:
    result = result * x

proc `@/`*[T](x, y: Matrix[T]): Matrix[T] {.noSideEffect.} =
  assert(x.M == y.M)
  assert(x.N == y.N)
  result = matrix[T](colMajor, x.M, y.N, newSeq[T](x.M * y.N))
  for i in 0..<x.M:
    for j in 0..<x.N:
      result[i, j] = x[i, j] * y[i, j]

proc reduce*[T, S](self: Matrix[T], init: S, f: (S, T, int, int) -> S):
    S {.noSideEffect.} =
  result = init
  for i in 0..<self.M:
    for j in 0..<self.N:
      result = f(result, self[i,j], i, j)

proc reduce*[T, S](self: Matrix[T], init: S, f: (S, T) -> S):
    S {.noSideEffect.} =
  self.reduce(init, (acc: S, val: T, row, col) => f(acc, val))

proc randomMatrix*[T: SomeReal](M, N: int, min: T = 0, max: T = 1, order = colMajor): Matrix[T] =
  result = matrix[T](order, M, N, newSeq[T](M * N))
  for i in 0 ..< (M * N):
    result.data[i] = random(max - min) + min

proc reduceRows*[T, S](self: Matrix[T], init: S, f: (S, T, int, int) -> S):
    Matrix[S] {.noSideEffect.} =
  result = matrix[S](self.order, self.M, 1, newSeq[S](self.M * self.N))
  for i in 0..<self.M:
    result[i,0] = init
    for j in 0..<self.N:
      result[i,0] = f(result[i,0], self[i,j], i, j)

proc reduceRows*[T, S](self: Matrix[T], init: S, f: (S, T) -> S):
    Matrix[S] {.noSideEffect.} =
  self.reduceRows(init, (acc: S, val: T, row, col) => f(acc, val))

proc transform*[T, S](self: Matrix[T], f: (T, int, int) -> S):
    Matrix[S] {.noSideEffect.} =
  result = matrix[S](colMajor, self.M, self.N, newSeq[S](self.M * self.N))
  for i in 0..<self.M:
    for j in 0..<self.N:
      result[i,j] = f(self[i,j], i, j)

proc transform*[T, S](self: Matrix[T], f: T -> S): Matrix[S] {.noSideEffect.} =
  self.transform((val: T, row, col) => f(val))

proc slice*[T](self: Matrix[T], lb, ub: int): Matrix[T] {.noSideEffect.} =
  # [lb, ub]
  result = matrix[T](self.order, ub - lb + 1, self.N, newSeq[T]((ub - lb + 1) * self.N))
  var pos = 0
  for i in lb..ub:
    for j in 0..<self.N:
      result[pos,j] = self[i,j]
    pos.inc
