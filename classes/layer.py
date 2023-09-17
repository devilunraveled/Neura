'''
    This is the class that deals with the arrangement
    of Neurons in a particular layer.
'''
from logs.logger import Logger
from .neuron import Neuron

class Layer:
    def __init__(self, layerIndex, layerSize = 8, domain = [0,1] ):
        self.index = layerIndex
        self.size = layerSize
        self.neurons = []

        self.logger = Logger()

        try:
            for neuronId in range (self.size):
                neuron = Neuron( layer=layerIndex, index=neuronId, domain = domain )
                self.neurons.append(neuron)
            self.logger.logInfo("Layer created.")
        except Exception as E:
            self.logger.logException(message=str(E))

    def updateLayer(self, newValues): # newValues is a vector of the same order.
        try:   
            if ( len( newValues ) != self.size ):
                raise Exception
            for neuronId in range ( self.size ):
                self.neurons[neuronId].setActivation( newValues[neuronId] )
            self.logger.logInfo("Layer Values updated.")
        except Exception as E:
            self.logger.logException(message=str(E))
