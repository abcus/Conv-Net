using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Fully_Connected_Layer {

        private int I_samples;
        private int previous_layer_size, layer_size;
        
        private bool needs_gradient;

        public Tensor I, B, W;

        // Tensors to hold ∂L/∂B and ∂L/∂W
        // Will have separate entries for each input sample
        public Tensor dB, dW;

        public Tensor V_dB, S_dB, V_dW, S_dW;

        public Fully_Connected_Layer(int previous_layer_size, int layer_size, bool needs_gradient) {
            this.previous_layer_size = previous_layer_size;
            this.layer_size = layer_size;
            this.needs_gradient = needs_gradient;

            this.B = new Tensor(1, this.layer_size);
            this.V_dB = new Tensor(1, this.layer_size);
            this.S_dB = new Tensor(1, this.layer_size);

            this.W = new Tensor(2, this.layer_size, this.previous_layer_size);
            this.V_dW = new Tensor(2, this.layer_size, this.previous_layer_size);
            this.S_dW = new Tensor(2, this.layer_size, this.previous_layer_size);

            // Biases and weights initialization
            // Biases are set to 0
            // Weights are set to random value from normal distribution * sqrt(2/previous layer size)
            for (int i = 0; i < layer_size; i++) {
                
                B.values[i] = 0.0;
                
                for (int j = 0; j < previous_layer_size; j++) {
                    this.W.values[i * this.previous_layer_size + j] = Utils.next_normal(Program.rand, 0, 1) * Math.Sqrt(2 / (Double)previous_layer_size);
                }
            }
        }
        public Tensor forward(Tensor I) {
            this.I = I;
            this.I_samples = I.dim_1;

            Tensor O = new Tensor(2, this.I_samples, this.layer_size);

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

            // Initialize ∂L/∂B and ∂L/∂W (have to store these for gradient descent)
            // Input samples is stored as the highest dimension to allow for faster access when calculating the sum across all input samples
            // Don't have to set values to 0.0 after updating because a new gradient tensor is created during each backward pass
            this.dB = new Tensor(2, this.I_samples, this.layer_size);
            this.dW = new Tensor(3, this.I_samples, this.layer_size, this.previous_layer_size);

            Parallel.For(0, this.I_samples, i => {
                for (int j = 0; j < layer_size; j++) {

                    // ∂L/∂B = ∂L/∂O * ∂O/∂B, stores it for gradient descent
                    this.dB.values[i * layer_size + j] = dO.values[i * layer_size + j]; // * 1

                    for (int k = 0; k < previous_layer_size; k++) {

                        // ∂L/∂W = ∂L/∂O * ∂O/∂W, stores it for gradient descent
                        this.dW.values[i * this.layer_size * this.previous_layer_size + j * this.previous_layer_size + k] = dO.values[i * layer_size + j] * this.I.values[i * previous_layer_size + k];
                    }
                }
            });

            // If not first layer and ∂L/∂I needs to be returned, calculate and return ∂L/∂I = ∂L/∂O * ∂O/∂I; otherwise return null
            if (this.needs_gradient == true) {
                Tensor dI = new Tensor(2, this.I_samples, this.previous_layer_size);
                
                Parallel.For(0, this.I_samples, i => {
                    for (int j = 0; j < this.previous_layer_size; j++) {
                        
                        Double dot_product = 0.0;
                        
                        for (int k = 0; k < this.layer_size; k++) {
                            
                            // W_transposed[j * layer_size + k] = W[k * previous_layer_size + j];
                            dot_product += (W.values[k * previous_layer_size + j] * dO.values[i * layer_size + k]);
                        }
                        dI.values[i * previous_layer_size + j] = dot_product;
                    }
                });
                return dI;
            } else {
                return null;
            }
        }
    }
}
