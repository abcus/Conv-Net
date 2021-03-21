using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Fully_Connected_Layer {

        private int input_samples, input_rows, input_columns, input_channels;
        private int previous_layer_size, layer_size;
        
        private bool needs_gradient;

        public Tensor input, biases, weights;

        // Tensors to hold dL/dB and dL/dW
        // Will have separate entries for each input sample
        public Tensor gradient_biases, gradient_weights;

        public Fully_Connected_Layer(int previous_layer_size, int layer_size, bool needs_gradient) {
            this.previous_layer_size = previous_layer_size;
            this.layer_size = layer_size;
            this.needs_gradient = needs_gradient;

            this.biases = new Tensor(1, this.layer_size, 1, 1, 1);
            this.weights = new Tensor(2, this.layer_size, this.previous_layer_size, 1, 1);

            // Biases and weights initialization
            // Biases are set to 0
            // Weights are set to random value from normal distribution * sqrt(2/previous layer size)
            for (int i = 0; i < layer_size; i++) {
                
                biases.values[i] = 0.0;
                
                for (int j = 0; j < previous_layer_size; j++) {
                    this.weights.values[i * this.previous_layer_size + j] = Program.normalDist.Sample() * Math.Sqrt(2 / (Double)previous_layer_size);
                }
            }
        }
        public Tensor forward(Tensor input) {
            this.input = input;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            Tensor output = new Tensor(2, input_samples, this.layer_size, 1, 1);

            // Selects the input sample from the batch
            Parallel.For(0, this.input_samples, i => {

                // Output is (dot product of input and corresponding weights) + bias
                for (int j = 0; j < this.layer_size; j++) {

                    Double sum = 0.0;
                    
                    for (int k = 0; k < this.previous_layer_size; k++) {
                        sum += input.values[i * previous_layer_size + k] * this.weights.values[j * previous_layer_size + k];
                    }
                    output.values[i * this.layer_size + j] = (sum + this.biases.values[j]);
                    sum = 0.0;
                }
            });
            return output;
        }

        public Tensor backward (Tensor gradient_output) {

            // Initialize dL/dB and dL/dW (have to store these for gradient descent)
            // Input samples is stored as the highest dimension to allow for faster access when calculating the sum across all input samples
            // Don't have to set values to 0.0 after updating because a new gradient tensor is created during each backward pass
            this.gradient_biases = new Tensor(2, this.layer_size, this.input_samples, 1, 1);
            this.gradient_weights = new Tensor(3, this.layer_size, this.previous_layer_size, this.input_samples, 1);

            Parallel.For(0, this.input_samples, i => {
                for (int j = 0; j < layer_size; j++) {

                    // dL/dB = dL/dO * dO/dB, stores it for gradient descent
                    this.gradient_biases.values[j * this.input_samples + i] = gradient_output.values[i * layer_size + j]; // * 1

                    for (int k = 0; k < previous_layer_size; k++) {

                        // dL/dW = dL/dO * dO/dW, stores it for gradient descent
                        this.gradient_weights.values[j * this.previous_layer_size * this.input_samples + k * this.input_samples + i] = gradient_output.values[i * layer_size + j] * this.input.values[i * previous_layer_size + k];
                    }
                }
            });
            
            // If not first layer and dL/dI needs to be returned, calculate and return dL/dI = dL/dO * dO/dI; otherwise return null
            if (this.needs_gradient == true) {
                Tensor gradient_input = new Tensor(2, this.input_samples, this.input_rows, this.input_columns, this.input_channels);
                Tensor transposed_weights_tensor = this.weights.transpose_2D();
                
                Parallel.For(0, this.input_samples, i => {
                    for (int j = 0; j < previous_layer_size; j++) {
                        
                        Double sum = 0.0;
                        
                        for (int k = 0; k < layer_size; k++) {
                            sum += (transposed_weights_tensor.values[j * layer_size + k] * gradient_output.values[i * layer_size + k]);
                        }
                        gradient_input.values[i * previous_layer_size + j] = sum;
                        sum = 0;
                    }
                });
                return gradient_input;
            } else {
                return null;
            }
        }
    }
}
