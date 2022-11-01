from math import *
from random import *

# Activation Functions
# Function is index 0, Derivative is index 1
sigmoid = [lambda x: 1 / (1 + exp(-x)), lambda x: (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))))]
relu = [lambda x: max(0, x), lambda x: 0 if x < 0 else 1]
none = [lambda x: x, lambda x: 1]

#Error Functions
ssr = [lambda pred, ans: sum([(ans[i] - pred[i]) ** 2 for i in range(len(pred))]), lambda pred, ans: [-2 * (ans[i] - pred[i]) for i in range(len(pred))]]


def gradientMean(gradients):
  ret = []
  for i in range(len(gradients[0])):
    temp1 = []
    for j in range(len(gradients[0][i])):
      temp2 = []
      for k in range(len(gradients[0][i][j])):
        total = [gradient[i][j][k] for gradient in gradients]
        temp2.append(sum(total) / len(gradients))
      temp1.append(temp2)
    ret.append(temp1)
  return ret

class nodeLayer:
    def __init__(this, numIn, width, actFunc=none, weights=False):
        this.numIn = numIn
        this.actFunc = actFunc
        this.weights = []
        if not weights:
            for i in range(width):
                temp = []
                for j in range(numIn + 1):
                    temp += [random() * 2 - 1]
                this.weights += [temp]
        else:
            this.weights = weights

    def feed(this, inp):
        inp.append(1)
        out = []
        for weights in this.weights:
            total = [inp[i] * weights[i] for i in range(this.numIn + 1)]
            sm = sum(total)
            out.append(this.actFunc[0](sm))
        inp.pop()
        return out

    def getWeightGradient(this, outputGradient, inp):
        inp.append(1)
        ret = []
        for i in range(len(this.weights)):
            weights = this.weights[i]
            temp = []
            total = [inp[x] * weights[x] for x in range(this.numIn + 1)]
            prev = outputGradient[i] * this.actFunc[1](sum(total))
            for j in range(len(weights)):
                temp.append(prev * inp[j])
            ret.append(temp)
        inp.pop()
        return ret

    def applyGradient(this, gradient, lr):
        for i in range(len(gradient)):
            for j in range(len(gradient[i])):
                this.weights[i][j] -= gradient[i][j] * lr

    def getInputGradient(this, outputGradient, inp):
        inp.append(1)
        ret = [0] * len(inp)
        for x in range(len(this.weights)):
            weights = this.weights[x]
            sm = sum([inp[i] * weights[i] for i in range(len(inp))])
            grad = outputGradient[x] * this.actFunc[1](sm)
            for i in range(len(inp)):
                ret[i] += grad * weights[i]
        inp.pop()
        return ret

class FNN:
  def __init__(this, numIn, dim, actFuncs, errFunc):
    this.numIn = numIn
    this.dim = dim
    this.layers = []
    this.errFunc = errFunc
    for i in range(len(dim)):
      nIn = dim[i - 1] if i > 0 else numIn
      lay = nodeLayer(nIn, dim[i], actFuncs[i])
      this.layers += [lay]
  
  def feed(this, inp):
    nextIn = inp
    for layer in this.layers:
      nextIn = layer.feed(nextIn)
    return nextIn

  def getWeightGradient(this, inp, ans):
    inputs = [inp]
    for layer in this.layers:
      inputs += [layer.feed(inputs[-1])]
    weightGradient = []
    errGrad = this.errFunc[1](inputs[-1], ans)
    for i in range(len(this.layers) - 1, -1, -1):
      layer = this.layers[i]
      outGrad = 0
      if len(weightGradient) == 0:
        outGrad = errGrad
      else:
        outGrad = weightGradient[0]
      weightGradient = [layer.getWeightGradient(outGrad, inputs[i])] + weightGradient
    return weightGradient

  def applyGradient(this, grad, lr):
    for i in range(len(this.layers)):
      layer = this.layers[i]
      layer.applyGradient(grad[i], lr)