import unittest
import nn

suite "layers":

  test "dense compute":
    var
      input = newMat[float](2, 3, @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      weights = newMat[float](3, 2, @[2.0, 4.0, 8.0, 3.0, 7.0, 2.0])
      dense = newDense(3, 2, weights)
      result = dense.compute(input)
      expected = newMat[float](2, 2, @[39.0, 16.0, 90.0, 43.0])
    check(result == expected)

  test "dense delta":
    var
      above = newMat[float](4, 2, @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      weights = newMat[float](3, 2, @[2.0, 4.0, 8.0, 3.0, 7.0, 2.0]) 
      dense = newDense(3, 2, weights)
      outgoing = newMat[float](3, 2)
      result = dense.delta(outgoing, above)
      expected = newMat[float](4, 3, @[10.0, 14.0, 11.0, 22.0, 36.0, 29.0, 34.0, 58.0, 47.0, 46.0, 80.0, 65.0])
    check(result == expected)

  test "sigmoid compute":
    var
      input = newMat[float](4, 2, @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      sigmoid = newSigmoid()
      result = sigmoid.compute(input)
      expected = newMat[float](4, 2, @[0.73105858, 0.88079708, 0.95257413, 0.98201379, 0.99330715, 0.99752738, 0.99908895, 0.99966465])
    check(result ~= expected)
