from nn import *

l = nodeLayer(1, 1)


def f(x): return 3 * x + 2


for i in range(100000):
    x = randint(-100, 100)
    output = l.feed([x])[0]
    error = (f(x) - output) ** 2
    print(error)
    l.train([-2 * (f(x) - output)], [x], .000001)
    print(.1 ** log(error))
