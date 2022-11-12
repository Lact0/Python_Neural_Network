from nn import *

lr = .01
n = 100
f = lambda x: sqrt(9 - (x - 3) ** 2)

net = loadFNN('stored.txt')

def trainBatch():
  gradients = []
  AvgErr = 0
  for i in range(n):
    x = 6 / n * i
    out = net.feed([x])
    AvgErr += ssr[0](out, [f(x)])
    gradient = net.getWeightGradient([x], [f(x)])
    gradients += [gradient]
  gradient = gradientMean(gradients)
  net.applyGradient(gradient, lr)
  print('Avg err: ', AvgErr / n)

def trainPoint(x):
  out = net.feed([x])
  err = ssr[0](out, [f(x)])
  gradient = net.getWeightGradient([x], [f(x)])
  net.applyGradient(gradient, lr)
  return err

def newNet():
  return FNN(1, [5, 5, 1], ['relu', 'relu', 'relu'], 'ssr')

def train(k = 100):
  for i in range(k):
    trainBatch()
    saveFNN('stored.txt', net)

def trainP(loops = 1):
  for j in range(loops):
    errors = []
    for i in range(n):
      errors.append(trainPoint(6 / n * i))
    print("Avg Err", sum(errors) / n)
    saveFNN('stored.txt', net)