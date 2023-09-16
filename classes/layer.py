'''
    This is the class that deals with the arrangement
    of Neurons in a particular layer.
'''
import sys
sys.path.append('/home/devilunraveled/Projects/NeuralNetworks/NN')


import defaults.py as env
from .neuron import Neuron

class Layer:
    def __init__(self, layerIndex, layerSize = env.DEFAULT_LAYER_SIZE, domain = [0,1] ):
        self.index = layerIndex
        self.size = layerSize
        self.neurons = []

        try:
            for neuronId in range (self.size):
                neuron = Neuron( neuronId, domain )
                self.neurons.append(neuron)
        except :
            print("Could not create Layer")

    def rectify(self, newValues): # newValues is a vector of the same order.
        try:   
            if ( len( newValues ) != self.size ):
                raise Exception
            for neuronId in range ( self.size ):
                self.neurons[neuronId].rectify( newValues[neuronId] );
        except:
            print("Could not modify layer values");
