using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class FullyConnectedLayer {

        private int previousLayerSize;
        private int layerSize;
        private bool needsGradient;
        public Double[][,,] weights;
        public Double[][,,] biases;
        public Double[][,,] gradientWeights;
        public Double[][,,] gradientBiases;
        public Double[,,] input;

        public Tensor biases_tensor;
        public Tensor weights_tensor;
        public Tensor input_tensor;

        public FullyConnectedLayer(int previousLayerSize, int layerSize, bool needsGradient) {
            this.previousLayerSize = previousLayerSize;
            this.layerSize = layerSize;
            this.needsGradient = needsGradient;
            this.weights = new Double[layerSize][,,];
            this.biases = new Double[layerSize][,,];
            this.gradientWeights = new Double[layerSize][,,];
            this.gradientBiases = new Double[layerSize][,,];

            for (int i = 0; i < layerSize; i++) {
                
                // Bias initialization (set to 0)
                Double[,,] tempBiases = new Double[1, 1, 1];
                tempBiases[0, 0, 0] = 0.0;
                this.biases[i] = tempBiases;

                // Weight initialization (set to random value from normal distribution * sqrt(2/previous layer size))
                Double[,,] tempWeights = new Double[1, 1, previousLayerSize];
                for (int j = 0; j < previousLayerSize; j++) {
                    tempWeights[0, 0, j] = Program.normalDist.Sample() * Math.Sqrt(2 / (Double)previousLayerSize);
                }
                this.weights[i] = tempWeights;

                // Initialize gradient of weights and biases with respect to loss (have to store these for gradient descent)
                Double[,,] tempWeightGradient = new Double[1, 1, previousLayerSize];
                this.gradientWeights[i] = tempWeightGradient;

                Double[,,] tempBiasGradient = new Double[1, 1, 1];
                this.gradientBiases[i] = tempBiasGradient;
            
            }
            this.biases_tensor = new Tensor(1, this.layerSize, 1, 1, 1);
            this.weights_tensor = new Tensor(2, this.layerSize, this.previousLayerSize, 1, 1);
            for (int i = 0; i < this.layerSize; i++) {
                for (int j=0; j < this.previousLayerSize; j++) {
                    this.weights_tensor.data[i * this.previousLayerSize + j] = this.weights[i][0, 0, j];
                }
            }
        }

        public Double[,,] forward(Double[,,] input) {
            this.input = input;
            Double[,,] output = new Double[1, 1, layerSize];

            // Output is dot product of input and corresponding weights + bias
            for (int i = 0; i < layerSize; i++) {
                output[0, 0, i] = Utils.dotProduct(input, weights[i]) + biases[i][0, 0, 0];
            }
            return output;
        }
        


        public Tensor forward_tensor(Tensor input) {
            this.input_tensor = input;
            Tensor output = new Tensor(2, input.num_samples, 1, 1, this.layerSize);

            for (int i=0; i < input.num_samples; i++) {
                for (int j=0; j < this.layerSize; j++) {

                    Double sum = 0.0;
                    for (int k=0; k < this.previousLayerSize; k++) {
                        sum += input.data[i * previousLayerSize + k] * this.weights_tensor.data[j * previousLayerSize + k];
                    }
                    output.data[i * this.layerSize + j] = (sum + this.biases_tensor.data[j]);
                }
            }
            return output;
        }


        public Double[,,] backward(Double[,,] gradientOutput) {

            Double[,,] gradientInput = new Double[1, 1, previousLayerSize];

            for (int i=0; i < layerSize; i++) {

                // dL/dB = dL/dO * dO/dB, stores it for gradient descent
                this.gradientBiases[i][0, 0, 0] += gradientOutput[0, 0, i] * 1;

                for (int j = 0; j < previousLayerSize; j++) {

                    // dL/dW = dL/dO * dO/dW, stores it for gradient descent
                    this.gradientWeights[i][0, 0, j] += gradientOutput[0, 0, i] * input[0, 0, j];
                }
            }

            // If gradient needed (i.e. not first layer) then return dL/dI = dL/dO * dO/dI; otherwise return null
            if (this.needsGradient == true) {
                for (int i = 0; i < previousLayerSize; i++) {
                    gradientInput[0, 0, i] = Utils.dotProduct(gradientOutput, Utils.transpose(this.weights)[i]);
                }
                return gradientInput;
            } else {
                return null;
            }
        }

        // Update weights and biases
        public void update (int batchSize) {
            for (int i = 0; i < layerSize; i ++) {
                this.biases[i][0,0,0] -= (this.gradientBiases[i][0,0,0] * Program.eta / batchSize);
                this.gradientBiases[i][0, 0, 0] = 0.0;

                for (int j=0; j < previousLayerSize; j++) {
                    this.weights[i][0, 0, j] -= (this.gradientWeights[i][0, 0, j] * Program.eta / batchSize);
                    this.gradientWeights[i][0, 0, j] = 0.0;
                }
            }
        }
    }
}
