var Vector = /** @class */ (function () {
    function Vector(entryCount) {
        var entryCountInteger = entryCount | 0; // remove fractional part //
        if (entryCountInteger < 0) {
            entryCountInteger = 0;
        }
        this.entries = new Array(entryCountInteger);
    }
    Vector.prototype.getIndex = function (entryIndex) {
        var entry = entryIndex | 0; // remove fractional part //
        if (entry < 0 || entry >= this.entries.length) {
            console.error("Vector index (" + entry + ") out of bounds");
            return -1;
        }
        return entry;
    };
    Vector.prototype.getValue = function (entryIndex) {
        var index = this.getIndex(entryIndex);
        if (index == -1) {
            return undefined;
        }
        return this.entries[index];
    };
    Vector.prototype.setValue = function (entryIndex, value) {
        var index = this.getIndex(entryIndex);
        if (index == -1) {
            return;
        }
        this.entries[index] = value;
    };
    Vector.prototype.getEntryCount = function () {
        return this.entries.length;
    };
    Vector.prototype.getEntryValues = function () {
        return this.entries.slice(); // creates a copy of the entries array //
    };
    return Vector;
}());
var Matrix = /** @class */ (function () {
    function Matrix(rowCount, columnCount) {
        this.rowCount = rowCount | 0;
        this.columnCount = columnCount | 0; // remove fractional parts //
        if (this.rowCount < 0) {
            this.rowCount = 0;
        }
        if (this.columnCount < 0) {
            this.columnCount = 0;
        }
        var entryCount = this.rowCount * this.columnCount;
        this.entries = new Array(entryCount);
    }
    Matrix.prototype.getIndex = function (rowIndex, columnIndex) {
        var row = rowIndex | 0;
        var column = columnIndex | 0; // remove fractional parts //
        if (row < 0 || row >= this.rowCount || column < 0 || column >= this.columnCount) {
            console.error("Matrix index (" + row + "," + column + ") out of bounds");
            return -1;
        }
        return row * this.columnCount + column;
    };
    Matrix.prototype.getValue = function (rowIndex, columnIndex) {
        var index = this.getIndex(rowIndex, columnIndex);
        if (index == -1) {
            return undefined;
        }
        return this.entries[index];
    };
    Matrix.prototype.setValue = function (rowIndex, columnIndex, value) {
        var index = this.getIndex(rowIndex, columnIndex);
        if (index == -1) {
            return;
        }
        this.entries[index] = value;
    };
    Matrix.prototype.getRowCount = function () {
        return this.rowCount;
    };
    Matrix.prototype.getColumnCount = function () {
        return this.columnCount;
    };
    Matrix.prototype.getEntryCount = function () {
        return this.entries.length;
    };
    return Matrix;
}());
var NeuralNetwork = /** @class */ (function () {
    function NeuralNetwork(inputLayerSize, outputLayerSize, hiddenLayerSizes) {
        if (hiddenLayerSizes === void 0) { hiddenLayerSizes = []; }
        var inputLayerSizeInteger = NeuralNetwork.getValidLayerSize(inputLayerSize);
        var outputLayerSizeInteger = NeuralNetwork.getValidLayerSize(outputLayerSize);
        this.outputs = new Array(hiddenLayerSizes.length + 2); // each layer of connected nodes //
        this.outputs[0] = new Vector(inputLayerSizeInteger);
        this.outputs[this.outputs.length - 1] = new Vector(outputLayerSizeInteger);
        for (var hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayerSizes.length; hiddenLayerIndex++) {
            var hiddenLayerSize = hiddenLayerSizes[hiddenLayerIndex];
            var hiddenLayerSizeInteger = NeuralNetwork.getValidLayerSize(hiddenLayerSize);
            this.outputs[hiddenLayerIndex + 1] = new Vector(hiddenLayerSizeInteger);
        }
        var temporaryStorageSize = inputLayerSizeInteger; // calculate size needed to copy largest layer //
        this.weights = new Array(hiddenLayerSizes.length + 1); // each connection between layers //
        this.biases = new Array(hiddenLayerSizes.length + 1);
        for (var layerIndex = 0; layerIndex < this.outputs.length - 1; layerIndex++) {
            var currentLayerSize = this.outputs[layerIndex].getEntryCount();
            var nextLayerSize = this.outputs[layerIndex + 1].getEntryCount();
            this.weights[layerIndex] = new Matrix(currentLayerSize, nextLayerSize);
            this.biases[layerIndex] = new Vector(nextLayerSize);
            if (temporaryStorageSize < nextLayerSize) {
                temporaryStorageSize = nextLayerSize;
            }
        }
        this.temporaryStorage = new Vector(temporaryStorageSize);
        this.randomizeWeightsAndBiases();
    }
    NeuralNetwork.getValidLayerSize = function (layerSize) {
        var layerSizeInteger = layerSize | 0; // remove fractional part //
        if (layerSizeInteger < 1) {
            layerSizeInteger = 1;
        }
        return layerSizeInteger;
    };
    NeuralNetwork.activationFunction = function (x) {
        return 1 / (1 + Math.exp(-x)); // sigmoid type activation function //
    };
    NeuralNetwork.convertActivationOutputToDerivate = function (x) {
        return x * (1 - x); // converts sigmoid output to its derivate value //
    };
    NeuralNetwork.prototype.randomizeWeightsAndBiases = function () {
        for (var _i = 0, _a = this.weights; _i < _a.length; _i++) { // weights ideally sampled from normal distribution, but uniform [-2,2) here //
            var weightMatrix = _a[_i];
            for (var rowIndex = 0; rowIndex < weightMatrix.getRowCount(); rowIndex++) {
                for (var columnIndex = 0; columnIndex < weightMatrix.getColumnCount(); columnIndex++) {
                    weightMatrix.setValue(rowIndex, columnIndex, 4 * Math.random() - 2);
                }
            }
        }
        for (var _b = 0, _c = this.biases; _b < _c.length; _b++) { // biases can be set to 0 //
            var biasVector = _c[_b];
            for (var entryIndex = 0; entryIndex < biasVector.getEntryCount(); entryIndex++) {
                biasVector.setValue(entryIndex, 0);
            }
        }
    };
    NeuralNetwork.prototype.clearLayerOutputs = function () {
        for (var _i = 0, _a = this.outputs; _i < _a.length; _i++) {
            var layer = _a[_i];
            for (var entryIndex = 0; entryIndex < layer.getEntryCount(); entryIndex++) {
                layer.setValue(entryIndex, undefined);
            }
        }
    };
    NeuralNetwork.prototype.setInputValues = function (inputValues) {
        var inputLayer = this.outputs[0];
        if (inputValues.length != inputLayer.getEntryCount()) {
            console.error("Provided input array does not match neural network dimensions");
            return;
        }
        for (var inputLayerIndex = 0; inputLayerIndex < inputLayer.getEntryCount(); inputLayerIndex++) {
            var inputValue = inputValues[inputLayerIndex];
            inputLayer.setValue(inputLayerIndex, inputValue);
        }
    };
    NeuralNetwork.prototype.getOutputValues = function () {
        var outputLayer = this.outputs[this.outputs.length - 1];
        return outputLayer.getEntryValues();
    };
    NeuralNetwork.prototype.calculateOutputs = function () {
        for (var layerIndex = 0; layerIndex < this.outputs.length - 1; layerIndex++) {
            var currentLayer = this.outputs[layerIndex];
            var nextLayer = this.outputs[layerIndex + 1];
            var weightMatrix = this.weights[layerIndex];
            var biasVector = this.biases[layerIndex];
            for (var nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
                var calculatedValue = biasVector.getValue(nextLayerIndex);
                for (var currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
                    calculatedValue += (currentLayer.getValue(currentLayerIndex) * weightMatrix.getValue(currentLayerIndex, nextLayerIndex));
                }
                nextLayer.setValue(nextLayerIndex, NeuralNetwork.activationFunction(calculatedValue));
            }
        }
    };
    NeuralNetwork.prototype.trainWeightsAndBiases = function (inputValues, expectedOutputValues) {
        this.setInputValues(inputValues);
        this.calculateOutputs();
        var outputLayer = this.outputs[this.outputs.length - 1];
        // calculate per-node error values on output layer and overwrite output layer values //
        for (var outputLayerIndex = 0; outputLayerIndex < outputLayer.getEntryCount(); outputLayerIndex++) {
            var expectedOutputValue = expectedOutputValues[outputLayerIndex];
            var actualOutputValue = outputLayer.getValue(outputLayerIndex);
            outputLayer.setValue(outputLayerIndex, (expectedOutputValue - actualOutputValue) *
                NeuralNetwork.convertActivationOutputToDerivate(actualOutputValue));
        }
        // iterate backwards through network with "current" layer starting one layer before output layer //
        for (var layerIndex = this.outputs.length - 2; layerIndex >= 0; layerIndex--) {
            var currentLayer = this.outputs[layerIndex];
            var nextLayer = this.outputs[layerIndex + 1];
            var weightMatrix = this.weights[layerIndex];
            var biasVector = this.biases[layerIndex];
            // calculate per-node error values for current layer and store in temporary memory //
            for (var currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
                var nodeError = 0;
                for (var nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
                    nodeError += weightMatrix.getValue(currentLayerIndex, nextLayerIndex) * nextLayer.getValue(nextLayerIndex);
                }
                nodeError *= NeuralNetwork.convertActivationOutputToDerivate(currentLayer.getValue(currentLayerIndex));
                this.temporaryStorage.setValue(currentLayerIndex, nodeError);
            }
            // correct weight values based on error //
            for (var currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
                var nodeValue = currentLayer.getValue(currentLayerIndex);
                for (var nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
                    var weightValue = weightMatrix.getValue(currentLayerIndex, nextLayerIndex);
                    var errorValue = nextLayer.getValue(nextLayerIndex);
                    weightValue += NeuralNetwork.learningRate * nodeValue * errorValue;
                    weightMatrix.setValue(currentLayerIndex, nextLayerIndex, weightValue);
                }
            }
            // correct bias values based on error //
            for (var nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
                var biasValue = biasVector.getValue(nextLayerIndex);
                var errorValue = nextLayer.getValue(nextLayerIndex);
                biasValue += NeuralNetwork.learningRate * errorValue;
                biasVector.setValue(nextLayerIndex, biasValue);
            }
            // replace current layer values with error values from temporary memory //
            for (var currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
                currentLayer.setValue(currentLayerIndex, this.temporaryStorage.getValue(currentLayerIndex));
            }
        }
        this.clearLayerOutputs();
    };
    NeuralNetwork.learningRate = 0.03;
    return NeuralNetwork;
}());
