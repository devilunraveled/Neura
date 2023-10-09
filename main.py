from classes import network

if __name__ == "__main__":
    numInputFeautures = int(input("Enter the number of input features : "))
    numOutputFeatures = int(input("Enter the number of output features : "))
    thisNeuralNetwork = network.NeuralNetwork(numInputFeautures = numInputFeautures, numOutputFeatures = numOutputFeatures)
