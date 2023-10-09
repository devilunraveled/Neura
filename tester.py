import numpy

from logs import logger

from classes import loss
from classes import neuron
from classes import layer
from classes import network

'''
    Given here are a set of functions that can be used to test the neural network.
    If any of these fail, then there is a possiblity of failure of the neural network.
'''

class Tester():
    def __init__(self):
        self.targetClass = None
        self.targetObject = None

        self.functionList = []
        self.argumentList = []

class TestLossMeanSquaredError(Tester):
    def __init__(self):
        self.targetClass = loss.MeanSquaredError
        self.targetObject = self.targetClass()
        
        if ( self.targetObject == None ):
            print(f"INIT ERROR-- Failed to create MeanSquaredError object.")
            return -1

        self.functionList = [self.targetObject.computeFunction, self.targetObject.computeDerivative]
        actual = numpy.array([1,2,5,6])
        predicted = numpy.array([1,2,5,7])

        self.validArgumentList = [[predicted, actual],[predicted, actual]]
        self.validExpectedOutput = [0.25, -0.5]
    
    def runAll(self):
        try :
            # Check if the corresponding function with corresponding valid arghument gives the valid output.
            for (thisFunction, arguments, expectedOutput) in zip(self.functionList, self.validArgumentList, self.validExpectedOutput):
                recievedOutput = thisFunction(*arguments)
                if ( type(recievedOutput) != type(expectedOutput) and recievedOutput != expectedOutput ):
                    print(f"TEST ERROR-- Failed to compute {thisFunction.__name__} function with parameters {arguments}.")
                    print(f"Expected output : {expectedOutput}, Received output : {recievedOutput}")
                    return -1

                print(f"TEST SUCCESS-- Class : MeanSquaredError Function : {thisFunction.__name__}.")
            return 0
        except Exception as E:
            print(f"TEST EXCEPTION -- {E}.")
            raise Exception


if __name__ == "__main__":
    tester = TestLossMeanSquaredError()
    tester.runAll()
