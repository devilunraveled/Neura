import math

class Activation :
    def __init__(self):
        pass

    def RELU(self, x):
        return max(0, x)

    def leakyRELU(self, x, gamma = 0.01):
        return max(gamma * x, x)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
