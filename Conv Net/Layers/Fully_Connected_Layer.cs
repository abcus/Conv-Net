using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Fully_Connected_Layer {

        private int previous_layer_size;
        private int layer_size;
        private bool needsGradient;
        public Double[][,,] weights;
        public Double[][,,] biases;
        public Double[][,,] gradientWeights;
        public Double[][,,] gradientBiases;
        public Double[,,] input;

        public Tensor biases_tensor;
        public Tensor weights_tensor;
        public Tensor gradient_biases_tensor;
        public Tensor gradient_weights_tensor;
        public Tensor input_tensor;

        public Fully_Connected_Layer(int previousLayerSize, int layerSize, bool needsGradient) {
            this.previous_layer_size = previousLayerSize;
            this.layer_size = layerSize;
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
            this.biases_tensor = new Tensor(1, this.layer_size, 1, 1, 1);
            this.weights_tensor = new Tensor(2, this.layer_size, this.previous_layer_size, 1, 1);
            for (int i = 0; i < this.layer_size; i++) {
                for (int j=0; j < this.previous_layer_size; j++) {
                    this.weights_tensor.values[i * this.previous_layer_size + j] = this.weights[i][0, 0, j];
                }
            }
        }
        public Tensor forward(Tensor input) {
            this.input_tensor = input;
            int sample_size = input.dim_1;

            this.gradient_biases_tensor = new Tensor(1, layer_size, 1, 1, 1);
            this.gradient_weights_tensor = new Tensor(2, layer_size, previous_layer_size, 1, 1);

            Tensor output = new Tensor(2, input.dim_1, this.layer_size, 1, 1);

            Parallel.For(0, input.dim_1, i => {

                // Output is dot product of input and corresponding weights + bias
                for (int j = 0; j < this.layer_size; j++) {

                    Double sum = 0.0;
                    for (int k = 0; k < this.previous_layer_size; k++) {
                        sum += input.values[i * previous_layer_size + k] * this.weights_tensor.values[j * previous_layer_size + k];
                    }
                    output.values[i * this.layer_size + j] = (sum + this.biases_tensor.values[j]);
                }
            });
            return output;
        }


        public Tensor backward (Tensor gradientOutput) {
            Tensor gradientInput = new Tensor(input_tensor.dimensions, input_tensor.dim_1, input_tensor.dim_2, input_tensor.dim_3, input_tensor.dim_4);

            // For each training example in the batch, gradient arrays are incremented (when updating, average gradients are calculated by dividing by batch size)
            // Keeping track of gradients for each training example separately requires another dimension on the gradient arrays, which is slower
            // Outer for loop not parallelized as locking arrays is slower
            for (int i=0; i < input_tensor.dim_1; i++) {
                for (int j = 0; j < layer_size; j++) {

                    // dL/dB = dL/dO * dO/dB, stores it for gradient descent
                    this.gradient_biases_tensor.values[j] += gradientOutput.values[i * layer_size + j]; // * 1
                    
                    for (int k = 0; k < previous_layer_size; k++) {

                        // dL/dW = dL/dO * dO/dW, stores it for gradient descent
                        this.gradient_weights_tensor.values[j * previous_layer_size + k] += gradientOutput.values[i * layer_size + j] * this.input_tensor.values[i * previous_layer_size + k];
                    }
                }
            }

            // If gradient needed (i.e. not first layer) then return dL/dI = dL/dO * dO/dI; otherwise return null
            if (this.needsGradient == true) {
                Tensor transposed_weights_tensor = weights_tensor.transpose_2D();
                Parallel.For(0, input_tensor.dim_1, i => {
                    for (int j = 0; j < previous_layer_size; j++) {
                        Double sum = 0.0;
                        for (int k = 0; k < layer_size; k++) {
                            sum += (transposed_weights_tensor.values[j * layer_size + k] * gradientOutput.values[i * layer_size + k]);
                        }
                        gradientInput.values[i * previous_layer_size + j] = sum;
                        sum = 0;
                    }
                });
                return gradientInput;
            } else {
                return null;
            }
        }

        /// <summary>
        /// Updates the biases and weights of the fully connected layer
        /// </summary>
        /// <param name="batch_size"></param>
        public void update (int batch_size) {
             
            for (int i = 0; i < layer_size; i++) {
                
                // bias gradient array contains sum of gradients from all examples in batch (divide by batch size to calculate the average gradient)
                this.biases_tensor.values[i] -= (this.gradient_biases_tensor.values[i] * Program.eta / batch_size);
                gradient_biases_tensor.values[i] = 0.0;

                for (int j=0; j < previous_layer_size; j++) {

                    // weights gradient array contains sum of gradients from all examples in batch (divide by batch size to calculate the average gradient)
                    this.weights_tensor.values[i * previous_layer_size + j] -= (gradient_weights_tensor.values[i * previous_layer_size + j] * Program.eta / batch_size);
                    gradient_weights_tensor.values[i * previous_layer_size + j] = 0.0;
                }
            }
        }
    }
}
