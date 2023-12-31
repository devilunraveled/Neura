from logs.logger import Logger

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
    def __init__( self, layer : int, index : int, domain = [0.0,1.0], activation = None, bias = 0.0, logger = None ):
        self.layer = layer
        if ( activation == None ):
            activation = 0.0
        self.activation = activation
        self.index = index
        self.bias = bias
        self.domain = domain
        
        if ( logger == None):
            logger = Logger()
        self.logger = logger
    
    def setActivation(self, newActivation):
        try :
            if ( newActivation > self.domain[0] and newActivation < self.domain[1] ):
                self.activation = newActivation
            else :
                self.logger.logWarning("Activation value is out of range. No change made.")
                self.logger.logWarning(f"Provided Activation: {newActivation}")
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
