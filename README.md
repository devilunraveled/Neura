# Neura
Neura is a raw python Neural Network architecture. With easy to configure ANN. The purpose of this project is **NOT** to replace the existing frameworks, rather to build conceptual clarity regarding th e low level functioning of a Deep Neural Network.


This project also deviates from standard practices at some points in the hope of encountering novel problems, whose solution also gives better conceptual clarity and a different set of performance benchmarks.

## Brief Documentation

### Neuron
- A `Neuron` is the most elementary structure of the _Neural Network_ . It is a node in a _Neural Network_.  
- Each `Neuron` has an _activation value_. This _activation value_ lies in the _domain_ of the `Neuron`, which is usually between 0 and 1.
- Each `Neuron`'s _activation value_ is a measure of how much the `Neuron` contributes to the _output_ of the _Neural Network_.

### Layer
- A `Layer` is a collection of `Neurons` in a _Neural Network_.
- Each `Layer` contains, in addition to the `Neurons`, a _unique_ `layerIndex` that is assigned to th e Layer at the time of creation.
- This `layerIndex` is used to identify the Layer in the _Neural Network_.
- The `inputFeatures` also comprise a layer, namely the `inputLayer`. 
- The `layerIndex` of the `inputLayer` is 0. The `outputLayer` has a `layerIndex` of 1.

### Neural Network
- A _Neural Network_ is a collection of _Layers_, and some methods that describe how the `Network` learns, defining operations such as `forwardPropagation`, `lossFunctions`, `intermediateActivationFunctions` and `finalActivationFunctions`.
- These functions can also be specified by the user, giving the user more control over the learning process.
