'''
    The overall structure of the neural network.
'''
from logs.logger import Logger

class NeuralNetwork:
    def __init__(self, inputFeautures : int , outputPossibilities : int, numHiddenLayers = 2, neuronVector = [16,16], domain = [0,1]):
        self.inputFeautures = inputFeautures
        self.outputPossibilities = outputPossibilities
        self.numHiddenLayers = numHiddenLayers
        self.neuronVector = neuronVector
        self.domain = domain

        self.logger = Logger()


    def train(self, targetEpochs = 50, hyperParameters : dict = {}, displayIntermediateResults : bool = False):
        for epoch in range(targetEpochs):
            self.__computeForwardPropagation()
            self.__computeLossFunction()
            self.__updateModelParameters(epoch, hyperParameters=hyperParameters)
            
            if ( displayIntermediateResults ):
                self.__displayModelResults()

    def __computeForwardPropagation(self):
        pass

    def __computeLossFunction(self):
        pass

    def __updateModelParameters(self, epoch = 50, hyperParameters = None):
        pass

    def __displayModelResults(self):
        pass
