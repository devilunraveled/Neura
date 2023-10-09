'''
    The overall structure of the neural network.
'''
from time import process_time
import numpy

from logs.logger import Logger
from .response import Success, Failure
from .layer import Layer

'''
    The main NeuralNetwork class
    the layer indexing includes all the layers.
    Input layer is indexed 0.
'''
class NeuralNetwork:
    def __init__(self, numInputFeautures : int , numOutputFeatures : int, 
                 numHiddenLayers = 2, layerVector = [16,16], domain = [0,1],
                 intermediateActivationFunction = None, endActivationFunction = None,
                 backPropagationAlgorithm = None, lossFunction = None):

        self.numInputFeautures = numInputFeautures
        self.inputFeautures = numpy.array(numInputFeautures)

        self.numOutputFeatures = numOutputFeatures
        self.outputFeatures = numpy.array(numOutputFeatures)
        
        self.numHiddenLayers = numHiddenLayers
        self.layers = numpy.array(numHiddenLayers + 2)
        self.layerVector = [numInputFeautures] + layerVector + [numOutputFeatures]
        
        self.weights = []

        self.domain = domain

        self.intermediateActivationFunction = self.__leakyRelu
        if ( intermediateActivationFunction != None ):
            self.intermediateActivationFunction = intermediateActivationFunction

        self.endActivationFunction = self.__softMax
        if ( endActivationFunction != None ):
            self.endActivationFunction = endActivationFunction
        
        self.backPropagationAlgorithm = self.__stochasticGradientDescent
        if ( backPropagationAlgorithm != None ):
            self.backPropagationAlgorithm = backPropagationAlgorithm
        
        self.lossFunction = self.__meanSquaredError
        if ( lossFunction != None ):
            self.lossFunction = lossFunction


        self.logger = Logger()
        
        self.__weights__ = False
        self.__layers__ = False 
    
    '''
        Fills the weights of the network
        with random values for initialization.
    '''
    def fillWeights(self):
        try :
            startTime = process_time()

            for layerIndex in range(self.layerVector.size - 1):
                self.weights.append(numpy.random.rand(self.layerVector[layerIndex], self.layerVector[layerIndex + 1]))
                self.__weights__ = True

            totalTime = process_time() - startTime

            return Success(time = totalTime)
        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()
    
    '''
        Initializes the layers of the network
    '''
    def initializeLayers(self, domain = [0,1], initialActivation = None):
        try :
            startTime = process_time()
            
            for layerIndex in range(self.layerVector.size):
                if ( initialActivation == None ):
                    thisActivation = random.random(domain[0], domain[1])
                else :
                    thisActivation = initialActivation

                self.layers[layerIndex] = Layer(layerIndex, self.layerVector[layerIndex], domain = domain, activation = thisActivation)
                self.__layers__ = True

            totalTime = process_time() - startTime
            return Success(time = totalTime)

        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()

    def feedInputFeatures(self, inputFeautures):
        try :
            self.inputFeautures = inputFeautures
            return Success(True)
        except Exception as E :
            self.logger.logException(message=str(E))
            return Failure()

    def train(self, targetEpochs = 50, hyperParameters : dict = {}, displayIntermediateResults : bool = False):
        try :
            startTime = process_time()
            
            epochTimes = []
            for epoch in range(targetEpochs):
                epochStartTime = process_time()

                self.__computeForwardPropagation()
                self.__computeLossFunction()
                self.__updateModelParameters(epoch, hyperParameters=hyperParameters)
                
                epochTime = process_time() - epochStartTime
                
                epochTimes.append(epochTime)

                if ( displayIntermediateResults ):
                    self.__displayModelResults()
                    self.logger.logInfo(f"Epoch {epoch + 1} completed. Model performace : {self.__getModelPerformance}")
                

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
    def __getWeights(self, layerIndex = 1):
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
    def __getLayerVector(self, layerIndex):
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
    def __getBiasVector(self, layerIndex ):
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
    def __computeForwardPropagation(self):
        try :
            startTime = process_time()

            self.logger.logInfo("Computing forward propagation.")

            for layerIndex in range(1,self.layers.size):
                
                layerValues = self.__getLayerVector(layerIndex - 1).returnObject
                weights = self.__getWeights(layerIndex).returnObject
                bias = self.__getBiasVector(layerIndex).returnObject
                
                if ( layerValues == None ):
                    raise Exception(f"Could not get values for layer {layerIndex - 1}.")
                if ( weights == None ):
                    raise Exception(f"Could not get weights for layer {layerIndex - 1}.")
                if ( bias == None ):
                    raise Exception(f"Could not get bias for layer {layerIndex - 1}.")
                
                # Updating the next in line layer.
                self.layers[layerIndex].updateLayer(weights * layerValues + bias)
            
            self.outputFeatures = self.layers[-1]
            self.logger.logInfo("Forward propagation completed.")
            totalTime = process_time() - startTime
            
            return Success(time = totalTime)

        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()

    def __computeLossFunction(self, actualValues = None, outputFeatures = None ):
        try :
            if ( actualValues is None or outputFeatures is None ):
                raise Exception("Loss function requires actual values and output features.")

            startTime = process_time()
            self.logger.logInfo("Computing loss function.")

            loss = self.lossFunction(actualValues, outputFeatures).returnObject

            totalTime = process_time() - startTime
            return Success(time = totalTime, returnObject = loss)

        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()
        

    def __updateModelParameters(self):
        pass

    def __displayModelResults(self):
        pass

    def __getModelPerformance(self):
        pass
    
    '''
        Computes the mean squared error between 
        the actual and predicted values.
    '''
    def __meanSquaredError(self, actualValues, predictedValues):
        try :
            if ( actualValues.shape() != predictedValues.shape() ):
                raise Exception("Shapes do not match for predicted and actual values.")
            
            startTime = process_time()
            
            self.logger.logInfo("Computing mean squared error.")
            errorVector =  (actualValues - predictedValues)**2

            totalTime = process_time() - startTime
            
            return Success(time = totalTime, returnObject = errorVector.mean())
        except Exception as E :
            self.logger.logException(message=str(E))
            return Failure()
    
    def __softMax(self, layerValues ):
        try :
            startTime = process_time()
            
            self.logger.logInfo("Computing softmax.")

            softmax = numpy.exp(layerValues) / numpy.sum(numpy.exp(layerValues))
            
            totalTime = process_time() - startTime
            
            return Success(time = totalTime, returnObject = softmax)
        except Exception as E:
            self.logger.logException(message=str(E))
            return Failure()

    def __stochasticGradientDescent(self):
        pass

    def __leakyRelu(self):
        pass
