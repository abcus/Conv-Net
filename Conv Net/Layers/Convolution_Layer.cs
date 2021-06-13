using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace Conv_Net {
    class Convolution_Layer : Base_Layer {

        public override bool trainable_parameters { get; }

        public int I_samples, I_rows, I_columns, I_channels;
        public int B_num;
        public int W_num, W_rows, W_columns, W_channels;

        public bool needs_gradient;
        public int pad_size;
        public int stride;
        public int dilation;
        public int groups;

        public Tensor I;
        public override Tensor B { get; set; }
        public override Tensor dB { get; set; }
        public override Tensor V_dB { get; set; }
        public override Tensor S_dB { get; set; }

        public override Tensor W { get; set; }
        public override Tensor dW { get; set; }
        public override Tensor V_dW { get; set; }
        public override Tensor S_dW { get; set; }

        public Convolution_Layer(int I_channels, int W_num, int W_rows, int W_columns, bool needs_gradient, int pad_size = 0, int stride = 1, int dilation = 1, int groups = 1) {

            this.trainable_parameters = true;

            this.I_channels = I_channels;
            this.B_num = W_num;
            this.W_num = W_num; this.W_rows = W_rows; this.W_columns = W_columns; this.W_channels = I_channels / groups;

            this.needs_gradient = needs_gradient;
            this.pad_size = pad_size;
            this.stride = stride;
            this.dilation = dilation;
            this.groups = groups;

            this.B = new Tensor(2, this.B_num, 1);
            this.dB = new Tensor(2, this.B_num, 1);
            this.V_dB = new Tensor(2, this.B_num, 1);
            this.S_dB = new Tensor(2, this.B_num, 1);

            this.W = new Tensor(4, this.W_num, this.W_rows, this.W_columns, this.W_channels);
            this.dW = new Tensor(4, this.W_num, this.W_rows, this.W_columns, this.W_channels);
            this.V_dW = new Tensor(4, this.W_num, this.W_rows, this.W_columns, this.W_channels);
            this.S_dW = new Tensor(4, this.W_num, this.W_rows, this.W_columns, this.W_channels);

            // Biases are initialized to 0
            // Weights are initialized to random value from normal distribution * sqrt(2/ (num_weights * weight_rows * weight_columns))
            for (int i = 0; i < B.values.Length; i++) { 
                B.values[i] = 0.0; 
            }
            for (int i=0; i < W.values.Length; i++) {
                this.W.values[i] = Utils.next_normal(Program.rand, 0, 1) * Math.Sqrt(2 / ((Double)this.W_num * this.W_rows * this.W_columns));
            }
        }

        public override Tensor forward(Tensor I) {

            this.I = I.pad(this.pad_size);
            
            // Set I_samples, I_rows, and I_columns after padding
            // Calculates O_samples, O_rows, O_columns, and O_channels
            this.I_samples = this.I.dim_1; this.I_rows = this.I.dim_2; this.I_columns = this.I.dim_3;
            int O_samples = this.I_samples;
            int O_rows = (this.I_rows - this.W_rows * this.dilation + this.dilation - 1) / this.stride + 1;
            int O_columns = (this.I_columns - this.W_columns * this.dilation + this.dilation - 1) / this.stride + 1;
            int O_channels = this.W_num;

            // If groups > 1, split I along channel dimension, and B and W along num dimension
            Tensor[] I_group = Utils.split(this.I, 4, this.groups);
            Tensor[] B_group = Utils.split(this.B, 1, this.groups);
            Tensor[] W_group = Utils.split(this.W, 1, this.groups);
            Tensor[] O_groups = new Tensor[this.groups];

            // For each tensor I, B, and W in the group, convert to matrix, calculate O, then convert back to tensor
            // O_matrix [O_channels x (O_sample * O_rows * O_columns)] = 
            //      W_matrix [W_num x (W_rows * W_columns * W_channels)] * 
            //      I_matrix [(W_rows * W_columns * W_channels) x (I_samples * O_rows * O_columns)] + 
            //      B_matrix [B_num x (I_samples * O_rows * O_columns)]
            for (int i = 0; i < this.groups; i++) {
                Tensor I_matrix = Utils.image_to_matrix(I_group[i], this.W_rows, this.W_columns, this.stride, this.dilation, Utils.OUTPUT_TYPE.OUTPUT);
                Tensor B_matrix = Utils.bias_to_matrix(B_group[i], this.I_samples, this.I_rows, this.I_columns, this.W_rows, this.W_columns, this.stride, this.dilation);
                Tensor W_matrix = Utils.kernel_to_matrix(W_group[i], Utils.OUTPUT_TYPE.OUTPUT);
                Tensor O_matrix = this.gemm_CPU(W_matrix, I_matrix, B_matrix);
                O_groups[i] = Utils.matrix_to_tensor(O_matrix, O_samples, O_rows, O_columns, O_channels / this.groups, Utils.OUTPUT_TYPE.OUTPUT);
            }
            // Merge the list of output tensors into a single tensor
            Tensor O = Utils.merge(O_groups, 4);
            return O;
        }
        
        public override Tensor backward(Tensor dO) {

            int dO_samples = dO.dim_1; int dO_rows = dO.dim_2; int dO_columns = dO.dim_3;

            // Calculate ∂L/∂B
            // ∂L/∂B is the convolution of ∂L/∂O over a tensor of 1s
            // dB [dB_num x 1] = 
            //      dO_matrix [dO_channels x (dO_sample * dO_rows * dO_columns)] * 
            //      column_1 [(dO_sample * dO_rows * dO_columns) x 1]
            Tensor dO_matrix_for_dB = Utils.kernel_to_matrix(dO, Utils.OUTPUT_TYPE.GRADIENT_BIAS);
            Tensor column_vector_1 = Utils.column_vector_1(dO_samples * dO_rows * dO_columns);
            this.dB = Utils.dgemm_cs(dO_matrix_for_dB, column_vector_1, this.dB);


            // Calculate ∂L/∂W
            // ∂L/∂W is the convolution of (∂L/∂O dilated by S) with stride D over I
            // dW_matrix [dW_num x (dW_rows * dW_columns * dW_channels)] =
            //      dO_matrix [dO_channels x (dO_sample * dO_rows * dO_columns)] * 
            //      I_matrix [(I_sample * dO_rows * dO_columns) x (W_rows * W_columns * W_channels)]      

            // If groups > 1, split dO and I along the channel dimension
            Tensor[] dO_group_for_dW = Utils.split(dO, 4, this.groups);
            Tensor[] I_group = Utils.split(this.I, 4, this.groups);
            Tensor[] dW_group = new Tensor[this.groups];

            // For each dO and I tensor in the group, convert to matrix, calculate dW, then convert back to tensor
            for (int i=0; i < this.groups; i++) {
                Tensor dO_matrix_for_dW = Utils.kernel_to_matrix(dO_group_for_dW[i], Utils.OUTPUT_TYPE.GRADIENT_WEIGHT);
                Tensor I_matrix = Utils.image_to_matrix(I_group[i], dO_rows, dO_columns, this.dilation, this.stride, Utils.OUTPUT_TYPE.GRADIENT_WEIGHT);
                Tensor dW_matrix = new Tensor(2, this.W_num / this.groups, this.W_rows * this.W_columns * this.W_channels);
                dW_matrix = this.gemm_CPU(dO_matrix_for_dW, I_matrix, dW_matrix);
                dW_group[i] = Utils.matrix_to_tensor(dW_matrix, this.W_num / this.groups, this.W_rows, this.W_columns, this.W_channels, Utils.OUTPUT_TYPE.GRADIENT_WEIGHT);
            }
            // Merge the list of weight gradient tensors into a single tensor
            this.dW = Utils.merge(dW_group, 1);
            this.I = null;

            // Calculate ∂L/∂I (if first layer, it is not needed and can return null)
            // ∂L/∂I is the full convolution of (180 rotated W dilated by D) over (∂L/∂O dilated by S and padded by (W_rows * D - D))
            // dI_matrix [dI_channels, (dI_samples * dI_rows * dI_columns)] = 
            //      W_rotated_matrix [W_channels x (W_num * W_rows * W_columns)] * 
            //      dO_dilated_padded_matrix [(W_num * W_rows * W_columns) x (I_samples * I_rows * I_columns)]
            if (this.needs_gradient == true) {

                // If groups > 1, split W and dO along the channel dimension
                Tensor[] W_group = Utils.split(this.W.rotate_180(), 1, this.groups);
                Tensor[] dO_group_for_dI = Utils.split(dO.dilate(this.stride).pad(this.W_rows * this.dilation - this.dilation).unpad(this.pad_size), 4, this.groups);
                Tensor[] dI_group = new Tensor[this.groups];

                // For each W and dO tensor in the group, convert to matrix, calculate dI, then convert back to tensor
                for (int i=0; i < this.groups; i++) {
                    // Size of dI is set to the size of I (before padding in the forward pass)
                    // Divide by groups
                    int dI_samples = this.I_samples; int dI_rows = this.I_rows - 2 * this.pad_size; int dI_columns = this.I_columns - 2 * this.pad_size; int dI_channels = this.I_channels / this.groups;

                    // For dO, dilate by S and pad for full convolution, then unpad by pad_size to avoid performing extra calculations, then convert to matrix
                    Tensor W_matrix = Utils.kernel_to_matrix(W_group[i], Utils.OUTPUT_TYPE.GRADIENT_INPUT);
                    Tensor dO_matrix_for_dI = Utils.image_to_matrix(dO_group_for_dI[i], this.W_rows, this.W_columns, 1, this.dilation, Utils.OUTPUT_TYPE.GRADIENT_INPUT);
                    Tensor dI_matrix = new Tensor(2, dI_channels, dI_samples * (dI_rows) * (dI_columns));
                    dI_matrix = this.gemm_CPU(W_matrix, dO_matrix_for_dI, dI_matrix);
                    dI_group[i] = Utils.matrix_to_tensor(dI_matrix, dI_samples, dI_rows, dI_columns, dI_channels, Utils.OUTPUT_TYPE.GRADIENT_INPUT);
                }
                // Merge the list of input gradient tensors into a single tensor
                Tensor dI = Utils.merge(dI_group, 4);                          
                return dI;
            } else {
                return null;
            }
        } 
        private Tensor gemm_CPU (Tensor A, Tensor B, Tensor C) {
            return Utils.dgemm_cs(A, B, C);
        }
        private Tensor gemm_GPU(Tensor A, Tensor B, Tensor C) {
            return null;
        }
    }
}
