# TypeScript Neural Network
An implementation of neural network maths in TypeScript.

### Motivation
In addition to solving classification problems like image categorization or speech recognition, neural networks can also simulate mathematical functions. In my opinion, the application of neural networks for "regression analysis" is an exciting field worth exploring. In this sample code, I show how a simple network of nodes, weights, and biases can be trained to approximate common operations on two numbers like averaging and multiplying.

### Project Layout
The __source__ folder contains the original, uncompiled TypeScript code. It represents my attempt at an object-oriented implementation of Vectors, Matrices, and Neural Networks. The __test__ folder contains a compiled, JavaScript version of the code along with an HTML page that can be opened in most browsers which demonstrates how to use the _NeuralNetwork_ class. The base folder contains a _tsconfig.json_ file in order to tell the TypeScript compiler how to generate the JavaScript file _script.js_.

### Compiling
I include the compiled JavaScript to ensure this step is optional. To compile on your own system, you would need a TypeScript compiler (tsc) whether it be from Node.js, a Microsoft IDE, or even a JavaScript version of the TypeScript compiler. Running the command "tsc" from the base folder should be sufficient.

### Resources Used
* https://adventuresinmachinelearning.com/neural-networks-tutorial/
* https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
