import numpy
from logs.logger import Logger

class Loss :
    def __init__(self):
        self.logger = Logger()
    
    def computeFunction(self, predicted : numpy.ndarray, actual : numpy.ndarray ):
        pass
    
    def computeDerivative(self, predicted : numpy.ndarray, actual : numpy.ndarray, pointOfDerivative : int ):
        pass

class MeanSquaredError(Loss):
    def __init__(self):
        pass

    def computeFunction(self, predicted : numpy.ndarray, actual : numpy.ndarray ):
        try :
            if ( actual.shape != predicted.shape ):
                self.logger.logError(message="Invalid shape provided as argument for MeanSquaredError.")
                return None

            return numpy.sum(numpy.square(actual - predicted)/actual.shape[0])
        except :
            self.logger.logException(message="MeanSquaredError Function")
            return None
    
    def computeDerivative(self, predicted : numpy.ndarray, actual : numpy.ndarray ):
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
