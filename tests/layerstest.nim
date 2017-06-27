import unittest
import nn

suite "layers":

  test "dense forward":
    var
      input = vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).asMatrix(2, 3, rowMajor)
      weights = vector([2.0, 4.0, 8.0, 3.0, 7.0, 2.0]).asMatrix(3, 2, rowMajor)
      dense = newDense(3, 2, weights)
      result = dense.forward(input)
      expected = vector([39.0, 16.0, 90.0, 43.0]).asMatrix(2, 2, rowMajor)
    check(result == expected)

  test "dense backward":
    var
      above = vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).asMatrix(4, 2, rowMajor)
      weights = vector([2.0, 4.0, 8.0, 3.0, 7.0, 2.0]).asMatrix(3, 2, rowMajor)
      dense = newDense(3, 2, weights)
      outgoing = matrix[float](colMajor, 3, 2, newSeq[float](3 * 2))
      result = dense.backward(outgoing, above)
      expected = vector(
        [10.0, 14.0, 11.0, 22.0, 36.0, 29.0, 34.0, 58.0, 47.0, 46.0, 80.0, 65.0]
      ).asMatrix(4, 3, rowMajor)
    check(result == expected)

  test "sigmoid forward":
    var
      input = vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).asMatrix(4, 2, rowMajor)
      sigmoid = newSigmoid()
      result = sigmoid.forward(input)
      expected = vector(
        [0.73105858, 0.88079708, 0.95257413, 0.98201379,
        0.99330715, 0.99752738, 0.99908895, 0.99966465]
      ).asMatrix(4, 2, rowMajor)
    check(result =~ expected)
