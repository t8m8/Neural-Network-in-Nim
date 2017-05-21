import unittest
import nn

import sequtils

suite "matrix":

  test "newMat":
    var mat = newMat[float](2, 3)
    check(mat.row == 2)
    check(mat.col == 3)

  test "newMat elm":
    var
      elm = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      mat = newMat[float](3, 2, elm)
    check(mat.row == 3)
    check(mat.col == 2)
    for i in 0..<mat.row:
      for j in 0..<mat.col:
        check(mat[i, j] == elm[i*mat.col+j])

  test "newMatRandom int":
    var
      (min, max) = (-100, 100)
      mat = newMatRandom[int](2, 3, min, max)
    for i in 0..<mat.row:
      for j in 0..<mat.col:
        check(min <= mat[i, j] and mat[i, j] < max)

  test "newMatRandom float":
    var
      (min, max) = (-100.0, 100.0)
      mat = newMatRandom[float](2, 3, min, max)
    for i in 0..<mat.row:
      for j in 0..<mat.col:
        check(min <= mat[i, j] and mat[i, j] < max)

  test "operator []=":
    var
      vec = @[1.0, 2.0, 3.0, 4.0, 5.0]
      mat = newMat[float](3, 5)
      ans = newMat[float](3, 5, sequtils.cycle(vec, 3))
    for i in 0..2:
      mat[i] = vec
    check(mat == ans)

  test "operator ==":
    var
      x = newMat[float](2, 2, @[1.0, 2.0, 3.0, 4.0])
      y1 = newMat[float](2, 2, @[1.0, 2.0, 3.0, 4.0])
      y2 = newMat[float](4, 1, @[1.0, 2.0, 3.0, 4.0])
      y3 = newMat[float](2, 2, @[1.0, 2.0, 3.0, 5.0])
    check(x == y1)
    check(x != y2)
    check(x != y3)

  test "operator *":
    var
      x = newMat[float](2, 2, @[1.0, -1.0, -2.0, 3.0])
      y = newMat[float](2, 2, @[1.0, 2.0, 3.0, 4.0])
      ans = newMat[float](2, 2, @[-2.0, -2.0, 7.0, 8.0])
    check(x * y == ans)

  test "transpose":
    var
      x = newMat[float](2, 3, @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      xt = newMat[float](3, 2, @[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    check(x.t() == xt)