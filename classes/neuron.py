from logs.logger import Log

'''
    This is the primitive class `Neuron`, 
    which is essentially the basic building
    block of the Neural Network.
'''
class Neuron:
    '''
        This is the base Neuron class.
        
        :param layer: Index of the layer this neuron belongs to.
        :param layerIndex: Index of the layer this neuron belongs to.
        :param domain: Domain of the neuron values.
        :param activation: Default activation of the neuron.
        :param bias: Default bias of the neuron.
    '''
    def __init__( self, layer : int, layerIndex : int, domain = [0,1], activation = 0.0, bias = 0.0 ):
        self.layer = layer
        self.activation = activation
        self.index = layerIndex
        self.bias = bias
        self.domain = domain

        self.logger = Log()
    
    def setActivation(self, newActivation):
        try :
            if ( newActivation < self.domain[0] or newActivation > self.domain[1] ):
                self.activation = newActivation
            else :
                self.logger.logWarning("Activation value is out of range. No change made.")
        except Exception as E :
            self.logger.logException(message=str(E))
    
    '''
        Updates the bias of the Neuron.
    '''
    def updateBias(self, newBias):
        self.bias = newBias
    
    '''
        Returns the index of the layer
    '''
    def getLayer(self):
        return self.layer
    
    '''
        Returns the index of the neuron within the layer.
    '''
    def getIndex(self):
        return self.index
    
    '''
        Returns the activation of the neuron.
    '''
    def getActivation(self):
        return self.activation
