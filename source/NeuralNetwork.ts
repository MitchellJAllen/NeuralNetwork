class NeuralNetwork {
	private outputs: Vector[]; // each node at each layer has an output value
	private weights: Matrix[];
	private biases: Vector[];

	private temporaryStorage: Vector;

	private static learningRate: number = 0.03;

	private static getValidLayerSize(layerSize: number): number {
		let layerSizeInteger = layerSize | 0; // remove fractional part //

		if (layerSizeInteger < 1) {
			layerSizeInteger = 1;
		}

		return layerSizeInteger;
	}

	private static activationFunction(x: number): number {
		return 1 / (1 + Math.exp(-x)); // sigmoid type activation function //
	}

	private static convertActivationOutputToDerivate(x: number): number {
		return x * (1 - x); // converts sigmoid output to its derivate value //
	}

	public constructor(inputLayerSize: number, outputLayerSize: number, hiddenLayerSizes: number[] = []) {
		let inputLayerSizeInteger = NeuralNetwork.getValidLayerSize(inputLayerSize);
		let outputLayerSizeInteger = NeuralNetwork.getValidLayerSize(outputLayerSize);

		this.outputs = new Array<Vector>(hiddenLayerSizes.length + 2); // each layer of connected nodes //

		this.outputs[0] = new Vector(inputLayerSizeInteger);
		this.outputs[this.outputs.length - 1] = new Vector(outputLayerSizeInteger);

		for (let hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayerSizes.length; hiddenLayerIndex++) {
			let hiddenLayerSize = hiddenLayerSizes[hiddenLayerIndex];
			let hiddenLayerSizeInteger = NeuralNetwork.getValidLayerSize(hiddenLayerSize);

			this.outputs[hiddenLayerIndex + 1] = new Vector(hiddenLayerSizeInteger);
		}

		let temporaryStorageSize = inputLayerSizeInteger; // calculate size needed to copy largest layer //

		this.weights = new Array<Matrix>(hiddenLayerSizes.length + 1); // each connection between layers //
		this.biases = new Array<Vector>(hiddenLayerSizes.length + 1);

		for (let layerIndex = 0; layerIndex < this.outputs.length - 1; layerIndex++) {
			let currentLayerSize = this.outputs[layerIndex].getEntryCount();
			let nextLayerSize = this.outputs[layerIndex + 1].getEntryCount();
	
			this.weights[layerIndex] = new Matrix(currentLayerSize, nextLayerSize);
			this.biases[layerIndex] = new Vector(nextLayerSize);

			if (temporaryStorageSize < nextLayerSize) {
				temporaryStorageSize = nextLayerSize;
			}
		}

		this.temporaryStorage = new Vector(temporaryStorageSize);

		this.randomizeWeightsAndBiases();
	}

	public randomizeWeightsAndBiases(): void { // starting point for a neural network //
		for (let weightMatrix of this.weights) { // weights ideally sampled from normal distribution, but uniform [-2,2) here //
			for (let rowIndex = 0; rowIndex < weightMatrix.getRowCount(); rowIndex++) {
				for (let columnIndex = 0; columnIndex < weightMatrix.getColumnCount(); columnIndex++) {
					weightMatrix.setValue(rowIndex, columnIndex, 4 * Math.random() - 2);
				}
			}
		}

		for (let biasVector of this.biases) { // biases can be set to 0 //
			for (let entryIndex = 0; entryIndex < biasVector.getEntryCount(); entryIndex++) {
				biasVector.setValue(entryIndex, 0);
			}
		}
	}

	public clearLayerOutputs(): void { // clears output values stored in each node of each layer //
		for (let layer of this.outputs) {
			for (let entryIndex = 0; entryIndex < layer.getEntryCount(); entryIndex++) {
				layer.setValue(entryIndex, undefined);
			}
		}
	}

	public setInputValues(inputValues: number[]): void {
		let inputLayer = this.outputs[0];

		if (inputValues.length != inputLayer.getEntryCount()) {
			console.error("Provided input array does not match neural network dimensions");

			return;
		}

		for (let inputLayerIndex = 0; inputLayerIndex < inputLayer.getEntryCount(); inputLayerIndex++) {
			let inputValue = inputValues[inputLayerIndex];

			inputLayer.setValue(inputLayerIndex, inputValue);
		}
	}

	public getOutputValues(): number[] {
		let outputLayer = this.outputs[this.outputs.length - 1];

		return outputLayer.getEntryValues();
	}

	public calculateOutputs(): void {
		for (let layerIndex = 0; layerIndex < this.outputs.length - 1; layerIndex++) {
			let currentLayer = this.outputs[layerIndex];
			let nextLayer = this.outputs[layerIndex + 1];

			let weightMatrix = this.weights[layerIndex];
			let biasVector = this.biases[layerIndex];

			for (let nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
				let calculatedValue = biasVector.getValue(nextLayerIndex);

				for (let currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
					calculatedValue += (
						currentLayer.getValue(currentLayerIndex) * weightMatrix.getValue(currentLayerIndex, nextLayerIndex)
					);
				}

				nextLayer.setValue(nextLayerIndex, NeuralNetwork.activationFunction(calculatedValue));
			}
		}
	}

	public trainWeightsAndBiases(inputValues, expectedOutputValues) {
		this.setInputValues(inputValues);
		this.calculateOutputs();

		let outputLayer = this.outputs[this.outputs.length - 1];

		// calculate per-node error values on output layer and overwrite output layer values //

		for (let outputLayerIndex = 0; outputLayerIndex < outputLayer.getEntryCount(); outputLayerIndex++) {
			let expectedOutputValue = expectedOutputValues[outputLayerIndex];
			let actualOutputValue = outputLayer.getValue(outputLayerIndex);

			outputLayer.setValue(outputLayerIndex, (expectedOutputValue - actualOutputValue) *
				NeuralNetwork.convertActivationOutputToDerivate(actualOutputValue)
			);
		}

		// iterate backwards through network with "current" layer starting one layer before output layer //

		for (let layerIndex = this.outputs.length - 2; layerIndex >= 0; layerIndex--) {
			let currentLayer = this.outputs[layerIndex];
			let nextLayer = this.outputs[layerIndex + 1];

			let weightMatrix = this.weights[layerIndex];
			let biasVector = this.biases[layerIndex];

			// calculate per-node error values for current layer and store in temporary memory //

			for (let currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
				let nodeError = 0;

				for (let nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
					nodeError += weightMatrix.getValue(currentLayerIndex, nextLayerIndex) * nextLayer.getValue(nextLayerIndex);
				}

				nodeError *= NeuralNetwork.convertActivationOutputToDerivate(currentLayer.getValue(currentLayerIndex));

				this.temporaryStorage.setValue(currentLayerIndex, nodeError);
			}

			// correct weight values based on error //

			for (let currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
				let nodeValue = currentLayer.getValue(currentLayerIndex);

				for (let nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
					let weightValue = weightMatrix.getValue(currentLayerIndex, nextLayerIndex);
					let errorValue = nextLayer.getValue(nextLayerIndex);

					weightValue += NeuralNetwork.learningRate * nodeValue * errorValue;

					weightMatrix.setValue(currentLayerIndex, nextLayerIndex, weightValue);
				}
			}

			// correct bias values based on error //

			for (let nextLayerIndex = 0; nextLayerIndex < nextLayer.getEntryCount(); nextLayerIndex++) {
				let biasValue = biasVector.getValue(nextLayerIndex);
				let errorValue = nextLayer.getValue(nextLayerIndex);

				biasValue += NeuralNetwork.learningRate * errorValue;

				biasVector.setValue(nextLayerIndex, biasValue);
			}

			// replace current layer values with error values from temporary memory //

			for (let currentLayerIndex = 0; currentLayerIndex < currentLayer.getEntryCount(); currentLayerIndex++) {
				currentLayer.setValue(currentLayerIndex, this.temporaryStorage.getValue(currentLayerIndex));
			}
		}

		this.clearLayerOutputs();
	}
}
