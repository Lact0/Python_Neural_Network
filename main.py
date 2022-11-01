from nn import *

l1 = nodeLayer(1, 20, sigmoid)
l2 = nodeLayer(20, 1, sigmoid)
lr = .01
n = 2
f = lambda x: sin(x)

net = FNN(1, [20, 1], [sigmoid, sigmoid], ssr)

def trainBatch():
  gradients = []
  AvgErr = 0
  for i in range(n):
    x = random() * 3.1415926535897 * 2
    gradient = []
    out1 = l1.feed([x])
    out2 = l2.feed(out1)
    outGrad = ssr[1](out2, [f(x)])
    AvgErr += ssr[0](out2, [f(x)])
    gradient += [l2.getWeightGradient(outGrad, out1)]
    outGrad = l2.getInputGradient(outGrad, out1)
    gradient = [l1.getWeightGradient(outGrad, [f(x)])] + gradient
    gradients += [gradient]
  gradient = gradientMean(gradients)
  l1.applyGradient(gradient[0], lr)
  l2.applyGradient(gradient[1], lr)
  print('Avg err: ', AvgErr / n)

def train():
  for i in range(100000):
    trainBatch()

train()