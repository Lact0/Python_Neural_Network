from math import *
from random import *

# Activation Functions
# Function is index 0, Derivative is index 1
sigmoid = [lambda x: 1 / (1 + exp(-x)), lambda x: (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))))]
relu = [lambda x: max(0, x), lambda x: 0 if x < 0 else 1]
none = [lambda x: x, lambda x: 1]


class nodeLayer:
    def __init__(this, numIn, width, actFunc=none, weights=False):
        this.numIn = numIn
        this.actFunc = actFunc
        this.weights = []
        if not weights:
            for i in range(width):
                temp = []
                for i in range(numIn + 1):
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
            return ret

    def train(this, outputGradient, inp, lr):
        gradient = this.getWeightGradient(outputGradient, inp)
        for i in range(len(gradient)):
            for j in range(len(gradient[i])):
                this.weights[i][j] -= gradient[i][j] * lr
