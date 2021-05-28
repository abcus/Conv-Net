using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Fully_Connected_Layer {

        private int I_samples, I_rows, I_columns, I_channels;
        private int previous_layer_size, layer_size;
        
        private bool needs_gradient;

        public Tensor I, B, W;

        // Tensors to hold dL/dB and dL/dW
        // Will have separate entries for each input sample
        public Tensor dB, dW;

        public Tensor V_dB, S_dB, V_dW, S_dW;

        public Fully_Connected_Layer(int previous_layer_size, int layer_size, bool needs_gradient) {
            this.previous_layer_size = previous_layer_size;
            this.layer_size = layer_size;
            this.needs_gradient = needs_gradient;

            this.B = new Tensor(1, this.layer_size, 1, 1, 1);
            this.V_dB = new Tensor(1, this.layer_size, 1, 1, 1);
            this.S_dB = new Tensor(1, this.layer_size, 1, 1, 1);

            this.W = new Tensor(2, this.layer_size, this.previous_layer_size, 1, 1);
            this.V_dW = new Tensor(2, this.layer_size, this.previous_layer_size, 1, 1);
            this.S_dW = new Tensor(2, this.layer_size, this.previous_layer_size, 1, 1);

            // Biases and weights initialization
            // Biases are set to 0
            // Weights are set to random value from normal distribution * sqrt(2/previous layer size)
            for (int i = 0; i < layer_size; i++) {
                
                B.values[i] = 0.0;
                
                for (int j = 0; j < previous_layer_size; j++) {
                    this.W.values[i * this.previous_layer_size + j] = Program.normalDist.Sample() * Math.Sqrt(2 / (Double)previous_layer_size);
                }
            }
        }
        public Tensor forward(Tensor I) {
            this.I = I;
            this.I_samples = I.dim_1;
            this.I_rows = I.dim_2;
            this.I_columns = I.dim_3;
            this.I_channels = I.dim_4;

            Tensor O = new Tensor(2, I_samples, this.layer_size, 1, 1);

            // Selects the input sample from the batch
            Parallel.For(0, this.I_samples, i => {

                // Output is (dot product of input and corresponding weights) + bias
                for (int j = 0; j < this.layer_size; j++) {

                    Double dot_product = 0.0;
                    
                    for (int k = 0; k < this.previous_layer_size; k++) {
                        dot_product += I.values[i * previous_layer_size + k] * this.W.values[j * previous_layer_size + k];
                    }
                    O.values[i * this.layer_size + j] = (dot_product + this.B.values[j]);
                }
            });
            return O;
        }

        public Tensor backward (Tensor dO) {

            // Initialize dL/dB and dL/dW (have to store these for gradient descent)
            // Input samples is stored as the highest dimension to allow for faster access when calculating the sum across all input samples
            // Don't have to set values to 0.0 after updating because a new gradient tensor is created during each backward pass
            this.dB = new Tensor(2, this.I_samples, this.layer_size);
            this.dW = new Tensor(3, this.I_samples, this.layer_size, this.previous_layer_size);

            Parallel.For(0, this.I_samples, i => {
                for (int j = 0; j < layer_size; j++) {

                    // dL/dB = dL/dO * dO/dB, stores it for gradient descent
                    this.dB.values[i * layer_size + j] = dO.values[i * layer_size + j]; // * 1

                    for (int k = 0; k < previous_layer_size; k++) {

                        // dL/dW = dL/dO * dO/dW, stores it for gradient descent
                        this.dW.values[i * this.layer_size * this.previous_layer_size + j * this.previous_layer_size + k] = dO.values[i * layer_size + j] * this.I.values[i * previous_layer_size + k];
                    }
                }
            });
            
            // If not first layer and dL/dI needs to be returned, calculate and return dL/dI = dL/dO * dO/dI; otherwise return null
            if (this.needs_gradient == true) {
                Tensor gradient_input = new Tensor(2, this.I_samples, this.I_rows, this.I_columns, this.I_channels);
                Tensor transposed_weights_tensor = this.W.transpose_2D();
                
                Parallel.For(0, this.I_samples, i => {
                    for (int j = 0; j < previous_layer_size; j++) {
                        
                        Double sum = 0.0;
                        
                        for (int k = 0; k < layer_size; k++) {
                            Console.WriteLine(transposed_weights_tensor.values[j * layer_size + k] - W.values[k * previous_layer_size + j]);
                            sum += (transposed_weights_tensor.values[j * layer_size + k] * dO.values[i * layer_size + k]);
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
