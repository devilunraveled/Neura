from typing_extensions import override
import numpy

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
    
    def areSame(self, objectA, objectB ):
        if type(objectA) != type(objectB) :
            print(f"TYPE ERROR-- {objectA} and {objectB} are not of the same type. {type(objectA)} and {type(objectB)}")
            return False

        if isinstance(objectA, (int, float, complex)):
            return objectA == objectB
        
        if isinstance(objectA, (list, tuple)):
            for (thisElementA, thisElementB) in zip(objectA, objectB):
                if not self.areSame(thisElementA, thisElementB):
                    return False
            return True


class TestLossMeanSquaredError(Tester):
    @override
    def __init__(self):
        actual = numpy.array([1,2,5,6])
        predicted = numpy.array([1,1,5,7])

        self.targetClass = loss.MeanSquaredError
        self.targetObject = self.targetClass(actual = actual)
        
        if ( self.targetObject == None ):
            print(f"INIT ERROR-- Failed to create MeanSquaredError object.")
            return -1

        self.functionList = [self.targetObject.computeFunction, self.targetObject.computeDerivative, self.targetObject.computeDerivative]
        self.validArgumentList = [[predicted],[3],[]]
        self.validExpectedOutput = [numpy.float64(0.5), numpy.float64(-0.5), numpy.array([0.0, 0.5, 0.0, -0.5])]
    
    def runAll(self):
        try :
            # Check if the corresponding function with corresponding valid argument gives the valid output.
            for (thisFunction, arguments, expectedOutput) in zip(self.functionList, self.validArgumentList, self.validExpectedOutput):
                recievedOutput = thisFunction(*arguments)

                if ( self.areSame(recievedOutput, expectedOutput) == False ):
                    print(f"TEST FAIL-- Incorrret Result for {thisFunction.__name__} function with parameters {arguments}.")
                    print(f"Expected output : {expectedOutput}, Received output : {recievedOutput}")
                
                print(f"TEST SUCCESS-- Class : MeanSquaredError Function : {thisFunction.__name__}.")
            return 0
        except Exception as E:
            print(f"TEST EXCEPTION -- {E}.")
            raise Exception


if __name__ == "__main__":
    tester = TestLossMeanSquaredError()
    tester.runAll()
