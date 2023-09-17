'''
    The overall structure of the neural network.
'''
from logs.logger import Logger
import numpy

class NeuralNetwork:
    def __init__(self, numInputFeautures : int , numOutputFeatures : int, 
                 numHiddenLayers = 2, neuronVector = [16,16], domain = [0,1],
                 intermediateActivationFunction = 'LeakyRelu', endActivationFunction = 'softmax',
                 backpropagationAlgorithm = 'SGD', lossFunction = 'crossEntropy'):

        self.numInputFeautures = numInputFeautures
        self.inputFeautures = numpy.array(numInputFeautures)

        self.numOutputFeatures = numOutputFeatures
        self.outputFeatures = numpy.array(numOutputFeatures)
        
        self.numHiddenLayers = numHiddenLayers
        self.neuronVector = neuronVector

        self.domain = domain
        self.intermediateActivationFunction = intermediateActivationFunction
        self.endActivationFunction = endActivationFunction
        self.backpropagationAlgorithm = backpropagationAlgorithm
        self.lossFunction = lossFunction

        self.logger = Logger()

    
    def feedInputFeatures(self, inputFeautures):
        self.inputFeautures = inputFeautures

    def train(self, targetEpochs = 50, hyperParameters : dict = {}, displayIntermediateResults : bool = False):
        try :
            for epoch in range(targetEpochs):
                self.__computeForwardPropagation()
                self.__computeLossFunction()
                self.__updateModelParameters(epoch, hyperParameters=hyperParameters)
                
                if ( displayIntermediateResults ):
                    self.__displayModelResults()

                self.logger.logInfo(f"Epoch {epoch + 1} completed. Model performace : {self.__getModelPerformance}")
        except Exception as E :
            self.logger.logException(message=str(E))
    
    def __computeForwardPropagation(self, batchSize = 10):
        pass

    def __computeLossFunction(self):
        pass

    def __updateModelParameters(self, epoch = 50, hyperParameters = None):
        pass

    def __displayModelResults(self):
        pass

    def __getModelPerformance(self):
        pass
