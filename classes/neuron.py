'''
    This is the primitive class `Neuron`, 
    which is essentially the basic building
    block of the Neural Network.
'''

class Neuron:
    '''
        This is the base Neuron class.
        
        :param layerIndex: Index of the layer this neuron belongs to.
        :param domain: Domain of the neuron values
    '''
    def __init__( self, layerIndex , domain = [0,1] ):
        self.layer = -1
        self.value = 0
        self.index = layerIndex
        self.bias = 0
        self.domain = domain
    
    '''
        Called after running the Backpropagation algorithm.
    '''
    def rectify( self , newValue ):
        try :
            if ( newValue < self.domain[0] or newValue > self.domain[1] ):
                raise Exception
            self.value = newValue
        except :
            print("Incorrect value passed.")
    
    def getLayer(self):
        return self.layer

    def getClassIndex(self):
        return self.index

    def getValue(self):
        return self.value

    def changeBias(self, newBias):
        self.bias = newBias
