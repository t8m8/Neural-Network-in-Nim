import nn

import os, strutils, future

proc loadImages(fname: string, size: int): Matrix[float] =
  var f: File = open(fname, FileMode.fmRead)
  defer: close(f)
  var
    images: Matrix[float] = newMat[float](size, 28 * 28)
    pos = 0
  while not f.endOfFile:
    images[pos] = f.readLine().split().map(parseFloat).map((val) => val / 255.0)
    pos.inc()
  result = images

proc loadLabels(fname: string, size: int): Matrix[float] =
  var f: File = open(fname, FileMode.fmRead)
  defer: close(f)
  var
    labels: Matrix[float] = newMat[float](size, 10)
    pos = 0
  while not f.endOfFile:
    labels[pos] = oneHot[float](10, parseInt(f.readLine()))
    pos.inc()
  result = labels

network:
  layers:
    Dense[28 * 28, 128]
    ReLU
    Dense[128, 128]
    ReLU
    Dense[128, 10]
    Softmax
  minimize: CrossEntropy
  optimize: SGD(0.5)

  echo "loading training data..."
  var
    trainInput = loadImages("../data/mnist/train-images.txt", 60000)
    trainOutput = loadLabels("../data/mnist/train-labels.txt", 60000)

    epochs = 3
    batchSize = 64
    options = newOptions(epochs, batchSize)

  options.addMeasure(newAccuracy())
  options.addMeasure(newMeanLoss())

  network.train(trainInput, trainOutput, options)

  echo "loading test data..."
  var
    testInput = loadImages("../data/mnist/test-images.txt", 10000)
    testOutput = loadLabels("../data/mnist/test-labels.txt", 10000)

  var
    probs = network.probs(testInput)
    meanloss = network.meanLossFromProbs(probs, testOutput)
    accuracy = network.accuracyFromProbs(probs, testOutput)
  echo "mean loss = " & $meanloss
  echo "accuracy = " & $accuracy