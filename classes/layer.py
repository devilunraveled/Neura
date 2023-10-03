'''
    This is the class that deals with the arrangement
    of Neurons in a particular layer.
'''
import numpy
from logs.logger import Logger
from .neuron import Neuron


class Layer:
    def __init__(self, layerIndex, layerSize = 8, domain = [0,1] ):
        self.index = layerIndex
        self.size = layerSize
        self.neurons = []
        self.__instantiated__ = False

        self.logger = Logger()

        try:
            for neuronId in range (self.size):
                neuron = Neuron( layer=layerIndex, index=neuronId, domain = domain )
                self.neurons.append(neuron)
            self.__instantiated__ = True
            self.logger.logInfo("Layer created.")
        except Exception as E:
            self.logger.logException(message=str(E))
    
    '''
        Updates the values of the 
        neurons in the layer.
    '''
    def updateLayer(self, newValues): # newValues is a vector of the same order.
        try:   
            if ( len( newValues ) != self.size ):
                self.logger.logException(message="Layer size mismatch.")
                raise Exception
            for neuronId in range ( self.size ):
                self.neurons[neuronId].setActivation( newValues[neuronId] )
            self.logger.logInfo("Layer Values updated.")
        except Exception as E:
            self.logger.logException(message=str(E))
    
    '''
        Returns the vector of activations
        of the neurons in the layer.
    '''
    def getVector(self):
        try :
            self.logger.logInfo(f"Getting activation vector for layer {self.index}")
            return numpy.array([neuron.activation for neuron in self.neurons])
        except Exception as E:
            self.logger.logException(message=str(E))
    
    '''
        Returns the vector of biases
        of the neurons in the layer.
    '''
    def getBiasVector(self):
        try :
            self.logger.logInfo(f"Getting bias vector for layer {self.index}")
            return numpy.array([neuron.bias for neuron in self.neurons])
        except Exception as E:
            self.logger.logException(message=str(E))
