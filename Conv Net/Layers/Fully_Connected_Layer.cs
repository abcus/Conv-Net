using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Conv_Net {
    class Fully_Connected_Layer : Base_Layer {

        private int previous_layer_size, layer_size;
        private bool needs_gradient;
        private int I_samples;

        public override bool trainable_parameters { get; }

        public Tensor I;
        public override Tensor B { get; set; }
        public override Tensor W { get; set; }
        public override Tensor dB { get; set; }
        public override Tensor dW { get; set; }
        public override Tensor V_dB { get; set; }
        public override Tensor V_dW { get; set; }
        public Tensor S_dB, S_dW;

        public Fully_Connected_Layer(int previous_layer_size, int layer_size, bool needs_gradient) {
            this.trainable_parameters = true;
            this.previous_layer_size = previous_layer_size;
            this.layer_size = layer_size;
            this.needs_gradient = needs_gradient;

            this.B = new Tensor(2, this.layer_size, 1);
            this.V_dB = new Tensor(2, this.layer_size, 1);
            this.S_dB = new Tensor(2, this.layer_size, 1);

            this.W = new Tensor(2, this.layer_size, this.previous_layer_size);
            this.V_dW = new Tensor(2, this.layer_size, this.previous_layer_size);
            this.S_dW = new Tensor(2, this.layer_size, this.previous_layer_size);

            // Biases and weights initialization
            // Biases are set to 0
            // Weights are set to random value from normal distribution * sqrt(2/previous layer size)
            for (int i=0; i < B.values.Length; i++) {
                B.values[i] = 0.0;
            }
            for (int i=0; i < W.values.Length; i++) {
                this.W.values[i] = Utils.next_normal(Program.rand, 0, 1) * Math.Sqrt(2 / (Double)previous_layer_size);
            }
        }
        public override Tensor forward(Tensor I) {
            this.I = I;
            this.I_samples = I.dim_1;

            // B_matrix [samples x layer_size] = 1_column [samples x 1] * transposed_B [1 x layer_size]
            // O [samples x layer_size] = I [samples x previous_layer_size] * W_transposed [previous_layer_size x layer_size] + B_matrix [samples x layer_size]
            Tensor B_matrix = new Tensor(2, this.I_samples, this.layer_size);
            B_matrix = Utils.dgemm_cs(Utils.column_vector_1(this.I_samples), this.B.transpose_2D(), B_matrix);
            Tensor O = new Tensor(2, this.I_samples, this.layer_size);
            O = Utils.dgemm_cs(I, this.W.transpose_2D(), B_matrix);
            return O;
        }

        public override Tensor backward (Tensor dO) {

            // Calculates ∂L/∂B and ∂L/∂W and stores these for gradient descent
            // Don't have to set values of dB and dW to 0.0 after updating because a new gradient tensor is created during each backward pass

            // dB [layer_size x 1] = dO_transposed [layer_size x sample] * 1_column [sample x 1]
            this.dB = new Tensor(2, this.layer_size, 1);
            this.dB = Utils.dgemm_cs(dO.transpose_2D(), Utils.column_vector_1(this.I_samples), this.dB);

            // dW [layer_size x previous_layer_size] = dO_transposed [layer_size x samples] * I [samples x previous_layer_size]
            this.dW = new Tensor(2, this.layer_size, this.previous_layer_size);
            this.dW = Utils.dgemm_cs(dO.transpose_2D(), this.I, this.dW);
            this.I = null;

            // Calculates ∂L/∂I (if first layer, it is not needed and can return null) 
            if (this.needs_gradient == true) {

                // dI [samples x previous_layer_size] = dO [samples x layer_size] * W [layer_size x previous_layer_size]
                Tensor dI = new Tensor(2, this.I_samples, this.previous_layer_size);
                dI = Utils.dgemm_cs(dO, this.W, dI);
                return dI;
            } else {
                return null;
            }
        }
    }
}
