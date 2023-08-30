'''
    This is the primitive class `Neuron`, 
    which is essentially the basic building
    block of the Neural Network.
'''

class Neuron:
    def __init__( self, layerIndex , domain = [0,1] ):
        self.layer = -1
        self.value = 0
        self.index = layerIndex
        self.bias = 0
        self.domain = domain

    def rectify( self , newValue ):
        try :
            if ( newValue < domain[0] or newValue > domain[1] ):
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
