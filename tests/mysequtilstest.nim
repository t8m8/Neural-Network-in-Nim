import unittest
import nn

suite "mysequtils":

  test "head":
    var
      vec0 = @[1.0, 2.0, 3.0]
      vec1 = @[5.0]
    check(vec0.head() == 1.0)
    check(vec1.head() == 5.0)

  test "last":
    var
      vec0 = @[1.0, 2.0, 3.0]
      vec1 = @[5.0]
    check(vec0.last() == 3.0)
    check(vec1.last() == 5.0)

  test "oneHot":
    var vec = oneHot[float](10, 5)
    check(vec.len == 10)
    for i in 0..<10:
      if i == 5:
        check(vec[i] == 1.0)
      else:
        check(vec[i] == 0.0)