using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class FullyConnectedLayer {

        private int previousLayerSize;
        private int layerSize;
        public Double[][,,] weights;
        public Double[][,,] biases;
        public Double[][,,] gradientWeights;
        public Double[][,,] gradientBiases;
        public Double[,,] input;

        public FullyConnectedLayer(int previousLayerSize, int layerSize) {
            this.previousLayerSize = previousLayerSize;
            this.layerSize = layerSize;
            this.weights = new Double[layerSize][,,];
            this.biases = new Double[layerSize][,,];
            this.gradientWeights = new Double[layerSize][,,];
            this.gradientBiases = new Double[layerSize][,,];

            // Weight initialization
            for (int i = 0; i < layerSize; i++) {
                Double[,,] tempWeights = new Double[1, 1, previousLayerSize];

                for (int j = 0; j < previousLayerSize; j++) {
                    // Set weight to random value from normal distribution * sqrt(2/previous layer size)
                    tempWeights[0, 0, j] = Program.normalDist.Sample() * Math.Sqrt(2 / (Double)previousLayerSize);
                }
                this.weights[i] = tempWeights;
            }

            // Bias initialization (set to 0)
            for (int i = 0; i < layerSize; i++) {
                Double[,,] tempBiases = new Double[1, 1, 1];
                tempBiases[0, 0, 0] = 0.0;
                this.biases[i] = tempBiases;
            }

            // Initialize gradient of weights and biases with respect to loss (have to store these for gradient descent)
            for (int i = 0; i < layerSize; i++) {
                Double[,,] tempWeightGradient = new Double[1, 1, previousLayerSize];
                this.gradientWeights[i] = tempWeightGradient;

                Double[,,] tempBiasGradient = new Double[1, 1, 1];
                this.gradientBiases[i] = tempBiasGradient;
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

        public Double[,,] backward(Double[,,] gradientOutput) {

            // Calculate gradients of loss with respect to weights and biases and stores them for gradient descent
            for (int i=0; i < layerSize; i++) {
                this.gradientBiases[i][0, 0, 0] = gradientOutput[0, 0, i] * 1;

                for (int j = 0; j < previousLayerSize; j++) {
                    this.gradientWeights[i][0, 0, j] = gradientOutput[0, 0, i] * input[0,0,j];
                }
            }

            // Calculate gradients of loss with respect to input, returns this value
            Double[,,] gradientInput = new Double[1, 1, previousLayerSize];
            for (int i = 0; i < previousLayerSize; i++) {
                gradientInput[0, 0, i] = Utils.dotProduct(gradientOutput, Utils.transpose(this.weights)[i]);
            }
            return gradientInput;
        }

        // Calculate gradients for all weights and biases, used in first fully connected layer (where backward is not called)
        public void storeGradient (Double[,,] inputGradient) {

            for (int i = 0; i < layerSize; i++) {
                this.gradientBiases[i][0, 0, 0] = inputGradient[0, 0, i] * 1;

                for (int j = 0; j < previousLayerSize; j++) {
                    this.gradientWeights[i][0, 0, j] = inputGradient[0, 0, i] * input[0, 0, j];
                }
            }
        }

        // Update weights and biases
        public void update () {
            for (int i = 0; i < layerSize; i ++) {
                this.biases[i][0,0,0] -= this.gradientBiases[i][0,0,0] * Program.eta;

                for (int j=0; j < previousLayerSize; j++) {
                    this.weights[i][0, 0, j] -= this.gradientWeights[i][0, 0, j] * Program.eta;
                }
            }
        }
    }
}
