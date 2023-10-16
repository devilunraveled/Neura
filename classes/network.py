'''
    The overall structure of the neural network.
'''
from time import process_time
import numpy
import random

from logs.logger import Logger
from .response import Success, Failure
from .layer import Layer
from .activations import Activation

'''
    The main NeuralNetwork class
    the layer indexing includes all the layers.
    Input layer is indexed 0.
'''
class NeuralNetwork:
    def __init__(self, numInputFeautures : int , numOutputFeatures : int, 
                 numHiddenLayers = 2, layerVector = [16,16], domain = [0,1],
                 intermediateActivationFunction = None, endActivationFunction = None,
                 backPropagationAlgorithm = None, lossFunction = None, logger = Logger()):

        self.numInputFeautures = numInputFeautures
        self.inputFeautures = numpy.array(numInputFeautures)

        self.numOutputFeatures = numOutputFeatures
        self.outputFeatures = numpy.array(numOutputFeatures)
        
        self.numHiddenLayers = numHiddenLayers
        
        self.layers = [Layer(-1, logger=logger)] * (self.numHiddenLayers + 2)
        self.layerActivation = [Activation()] * (self.numHiddenLayers + 2)
        self.layerVector = [numInputFeautures] + layerVector + [numOutputFeatures]
        
        self.weights = []

        self.domain = domain

        self.logger = logger
        
        self.__weights__ = False
        self.__layers__ = False 
    
    '''
        Fills the weights of the network
        with random values for initialization.
    '''
    def fillWeights(self):
        try :
            startTime = process_time()

            for layerIndex in range(len(self.layerVector) - 1):
                self.weights.append(numpy.random.rand(self.layerVector[layerIndex], self.layerVector[layerIndex + 1]))
            
            self.__weights__ = True
            
            totalTime = process_time() - startTime

            print(self.weights[0])
            return Success(time = totalTime)
        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()
    
    '''
        Initializes the layers of the network
    '''
    def initializeLayers(self, domain = [0.0,1.0], initialActivation = None):
        try :
            startTime = process_time()
            
            for layerIndex in range(len(self.layerVector)):
                if ( initialActivation == None ):
                    thisActivation = random.randrange(domain[0], domain[1])
                else :
                    thisActivation = initialActivation
                
                self.layers[layerIndex] = Layer(layerIndex, self.layerVector[layerIndex], domain = domain, activation = thisActivation, logger= self.logger)
                self.__layers__ = True

            totalTime = process_time() - startTime
            return Success(time = totalTime)

        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()

    def feedInputFeatures(self, inputFeautures):
        try :
            self.inputFeautures = inputFeautures
            self.layers[0].setVector(self.inputFeautures)
            return Success(True)
        except Exception as E :
            self.logger.logException(message=str(E))
            return Failure()

    def train(self, targetEpochs = 1, hyperParameters : dict = {}, displayIntermediateResults : bool = False):
        try :
            startTime = process_time()
            
            self.logger.logInfo("Training started.")
            epochTimes = []
            for epoch in range(targetEpochs):
                epochStartTime = process_time()

                self.computeForwardPropagation()
                # self.__computeLossFunction()
                # self.__updateModelParameters()
                
                print(f"Epoch {epoch + 1} : ")
                
                for layerIndex, layer in enumerate(self.layers):
                    print(f"Layer {layerIndex} : {layer}")

                epochTime = process_time() - epochStartTime
                
                epochTimes.append(epochTime)

                # if ( displayIntermediateResults ):
                    # self.__displayModelResults()
                    # self.logger.logInfo(f"Epoch {epoch + 1} completed. Model performace : {self.__getModelPerformance}")
                

            totalTime = process_time() - startTime
            return Success(time = totalTime, returnObject = {'epochList' : epochTimes})
        except Exception as E :
            self.logger.logException(message=str(E))
            return Failure()
    
    def __refreshWeights(self):
        try :
            startTime = process_time()

            if ( self.__weights__ == False ):
                self.fillWeights()
                totalTime = process_time() - startTime
                return Success(time = totalTime)
            else :
                return Success(lazy = True)
        except Exception as E :
            self.logger.logException(message=str(E))

    '''
        Retrieves the weights preceding a 
        particular layer in the neural network.
    '''
    def __getWeights(self, layerIndex : int):
        try :
            if ( layerIndex < 0 or layerIndex > self.numHiddenLayers + 2 ):
                self.logger.logError(message="Invalid layer index.")
                return Failure(message = "Invalid layer index")

            startTime = process_time()

            self.__refreshWeights()
            layerWeights = self.weights[layerIndex]
            totalTime = process_time() - startTime

            return Success(time = totalTime, returnObject = layerWeights)
        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()
    
    '''
        Retrieves the vector of the
        activation values of a particular layer.
    '''
    def __getLayerVector(self, layerIndex : int):
        try :
            if ( layerIndex < 0 or layerIndex > self.numHiddenLayers + 2 ):
                self.logger.logError(message="Invalid layer index.")
                return Failure(message = "Invalid layer index")

            startTime = process_time()

            thisLayer = self.layers[layerIndex]
            layerVector = thisLayer.getVector()
            
            totalTime = process_time() - startTime

            return Success(time = totalTime, returnObject = layerVector)
            
        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()
    
    '''
        Retrieves the vector of the
        bias values of a particular layer.
    '''
    def __getBiasVector(self, layerIndex : int ):
        try :
            if ( layerIndex < 0 or layerIndex > self.numHiddenLayers + 2 ):
                self.logger.logException(message="Invalid layer index.")
                return Failure(message = "Invalid layer index")
            
            startTime = process_time()

            thisLayer = self.layers[layerIndex]
            biasVector = thisLayer.getBiasVector()

            totalTime = process_time() - startTime

            return Success(time = totalTime, returnObject = biasVector)
        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()
    '''
        Compute the forward propagation for a 
        single pass of the neural network.
    '''
    def computeForwardPropagation(self):
        try :
            startTime = process_time()

            self.logger.logInfo("Computing forward propagation.")

            for layerIndex in range(1,len(self.layers)):
                
                layerValues = self.__getLayerVector(layerIndex - 1).returnObject
                weights = self.__getWeights(layerIndex - 1).returnObject
                bias = self.__getBiasVector(layerIndex).returnObject
                
                if ( layerValues is None ):
                    raise Exception(f"Could not get values for layer {layerIndex - 1}.")
                if ( weights is None ):
                    raise Exception(f"Could not get weights for layer {layerIndex - 1}.")
                if ( bias is None ):
                    raise Exception(f"Could not get bias for layer {layerIndex - 1}.")
                
                print(weights.T)
                print(layerValues)
                # Updating the next in line layer.
                if ( self.layers[layerIndex] is None ):
                    self.logger.logWarning(f"Layer {layerIndex} not initialized.")
                    return Failure()
                
                newLayer = weights.T @ layerValues + bias
                normalizedLayer = [self.layerActivation[layerIndex].sigmoid(x) for x in newLayer]

                self.layers[layerIndex].updateLayer(normalizedLayer)
            
            self.outputFeatures = self.layers[-1]
            self.logger.logInfo("Forward propagation completed.")
            totalTime = process_time() - startTime
            
            return Success(time = totalTime)

        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()

    def __updateModelParameters(self):
        pass

    def __displayModelResults(self):
        pass

    def __getModelPerformance(self):
        pass
    
