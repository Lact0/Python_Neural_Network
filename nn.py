from math import *

#Activation Functions
#Function is index 0, Derivative is index 1
sigmoid = [lambda x: 1 / (1 + exp(-x)), lambda x: (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))))]
relu = [lambda x: max(0, x), lambda x: 0 if x < 0 else 1]

class layer:
    def __init__(self, width, actFunc):
