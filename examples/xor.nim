import nn
import random

proc genXorData(n: int): (Matrix[float], Matrix[float]) =
  var (input, output) = (newMat[float](n, 2), newMat[float](n, 2))
  for i in 0..<n:
    input[i, 0] = random(2).float
    input[i, 1] = random(2).float
    output[i] = oneHot[float](2, input[i, 0].int xor input[i, 1].int)
  result = (input, output)

network:
  layers:
    Dense[2, 10]
    Sigmoid
    Dense[10, 2]
    Softmax
  minimize: CrossEntropy
  optimize: SGD(0.1)

  var
    (trainInput, trainOutput) = genXorData(1000000)
    epochs = 3
    batchSize = 64
    options = newOptions(epochs, batchSize)

  options.addMeasure(newAccuracy())
  options.addMeasure(newMeanLoss())

  network.train(trainInput, trainOutput, options)

  var
    testData = concat(@[0.0, 0.0], @[0.0, 1.0], @[1.0, 1.0], @[1.0, 0.0])
    testInput = newMat[float](4, 2, testData)
    res = network.predict(testInput)

  for i in 0..<testInput.row:
    echo $testInput[i,0] & " xor " & $testInput[i,1] & " = " & $res[i,0]