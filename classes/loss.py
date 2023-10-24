from typing import Union
from typing_extensions import override
import numpy
from logs.logger import Logger

class Loss :
    def __init__(self, logger = None):
        self.logger = logger if logger != None else Logger() 
        self.actual = None
    
    def computeFunction( self, predicted : numpy.ndarray ):
        self.predicted = predicted

        if ( self.actual is None ):
            self.logger.logError(message="No actual data provided for MeanSquaredError.")
            return None
        
        if ( self.actual.shape != self.predicted.shape ):
            self.logger.logError(message="Invalid shape provided as argument for MeanSquaredError.")
            return None

        pass
    
    def computeDerivative( self ):
        if ( hasattr(self, "predicted") == False ):
            self.logger.logError(message="No predicted data provided for MeanSquaredError derivative.")
            return False
        
        if ( self.actual is None or self.predicted is None ):
            self.logger.logError(message=f"No actual or predicted data provided for MeanSquaredError derivative. Actual : {self.actual}, Predicted : {self.predicted}")
            return False

        if ( self.actual.shape != self.predicted.shape ):
            self.logger.logError(message="Invalid shape provided as argument for MeanSquaredError derivative.")
            return False
        
class MeanSquaredError(Loss):
    def __init__(self, actual : numpy.ndarray, logger = None):
        super().__init__(logger = logger)
        self.actual = actual
    
    @override
    def computeFunction(self, predicted : numpy.ndarray):
        try :
            super().computeFunction(predicted)

            return numpy.sum(numpy.square(self.actual - self.predicted)/self.actual.shape[0])
        except :
            self.logger.logException(message="MeanSquaredError Function")
            return None
    
    @override
    def computeDerivative(self, neuronPosition : Union[int, numpy.ndarray] = -1):
        try :
            super().computeDerivative()
            
            if ( neuronPosition == -1 ):
                return 2*(self.actual - self.predicted)/self.actual.shape[0]
            else :
                return 2*(self.actual[neuronPosition] - self.predicted[neuronPosition])/self.actual.shape[0]
        except :
            self.logger.logException(message="MeanSquaredError Derivative")
            return None

if __name__ == "__main__":
    predicted = numpy.array([1,2,4,4,5])
    actual = numpy.array([1,2,3,4,5])
    meanSquared = MeanSquaredError(actual)
    
    print(meanSquared.computeFunction(predicted = predicted))
    print(meanSquared.computeDerivative(neuronPosition = 0))
