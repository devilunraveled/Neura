from typing_extensions import override
import numpy
from logs.logger import Logger

class Loss :
    def __init__(self, logger = None):
        self.logger = logger if logger != None else Logger() 
    def computeFunction( self ):
        pass
    
    def computeDerivative( self ):
        pass

class MeanSquaredError(Loss):
    def __init__(self, predicted : numpy.ndarray, actual : numpy.ndarray, logger = None):
        super().__init__(logger = logger)
        self.predicted = predicted
        self.actual = actual
    
        if ( self.actual.shape != self.predicted.shape ):
            self.logger.logWarning(message="Shape mismatch for Mean Squared Error Calculation.")
    @override
    def computeFunction(self):
        try :
            if ( self.actual.shape != self.predicted.shape ):
                self.logger.logError(message="Invalid shape provided as argument for MeanSquaredError.")
                return None

            return numpy.sum(numpy.square(actual - predicted)/actual.shape[0])
        except :
            self.logger.logException(message="MeanSquaredError Function")
            return None
    
    @override
    def computeDerivative(self):
        try :
            if ( actual.shape != predicted.shape ):
                self.logger.logError(message="Invalid shape provided as argument for MeanSquaredError derivative.")
                return None
            return numpy.sum(2*(actual - predicted)/actual.shape[0])
        except :
            self.logger.logException(message="MeanSquaredError Derivative")
            return None

if __name__ == "__main__":
    meanSquared = MeanSquaredError()
    predicted = numpy.array([1,2,4,4,5])
    actual = numpy.array([1,2,3,4,5])

    print(meanSquared.computeFunction(predicted, actual))
    print(meanSquared.computeDerivative(predicted, actual))
