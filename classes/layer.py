'''
    This is the class that deals with the arrangement
    of Neurons in a particular layer.
'''

class Layer:
    def __init__(self, layerIndex, layerSize = DEFAULT_LAYER_SIZE, domain = [0,1] ):
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
