from classes import network
import numpy

if __name__ == "__main__":
    numInputFeautures = int(input("Enter the number of input features : "))
    inputFeautures = numpy.random.rand(numInputFeautures)
    numOutputFeatures = int(input("Enter the number of output features : "))
    
    thisNeuralNetwork = network.NeuralNetwork(numInputFeautures = numInputFeautures, numOutputFeatures = numOutputFeatures)
    thisNeuralNetwork.initializeLayers()
    thisNeuralNetwork.feedInputFeatures(inputFeautures)
    thisNeuralNetwork.fillWeights()
    thisNeuralNetwork.train()
    # Generate random input features
