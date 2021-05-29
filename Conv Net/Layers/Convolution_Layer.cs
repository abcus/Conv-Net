using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace Conv_Net {
    class Convolution_Layer {

        public int I_samples, I_rows, I_columns, I_channels;
        public int B_num;
        public int F_num, F_rows, F_columns, F_channels;
        public int O_samples, O_rows, O_columns, O_channels;

        public int dI_samples, dI_rows, dI_columns, dI_channels;
        public int dB_num;
        public int dF_num, dF_rows, dF_columns, dF_channels;
        public int dO_rows, dO_columns;

        public int pad_size;
        public bool needs_gradient;
        public  int stride;
        public int dilation;

        // Input, bias, and filter tensors
        public Tensor I, B, F;

        // Tensors to hold ∂L/∂B and ∂L/∂F
        // Separate entries for each input sample to allow for multi-threading
        public Tensor dB, dF;
        public Tensor V_dB, S_dB, V_dF, S_dF;

        public Convolution_Layer(int I_channels, int F_num, int F_rows, int F_columns, bool needs_gradient, int pad_size = 0, int stride = 1, int dilation = 1) {

            this.I_channels = I_channels;
            this.B_num = F_num;
            this.F_num = F_num; this.F_rows = F_rows; this.F_columns = F_columns; this.F_channels = I_channels;

            this.dB_num = this.B_num;
            this.dF_num = this.F_num; this.dF_rows = this.F_rows; this.dF_columns = this.F_columns; this.dF_channels = this.F_channels;

            this.needs_gradient = needs_gradient;
            this.pad_size = pad_size;
            this.stride = stride;
            this.dilation = dilation;

            this.B = new Tensor(1, this.B_num);
            this.V_dB = new Tensor(1, this.dB_num);
            this.S_dB = new Tensor(1, this.dB_num);

            this.F = new Tensor(4, this.F_num, this.F_rows, this.F_columns, this.F_channels);
            this.V_dF = new Tensor(4, this.dF_num, this.dF_rows, this.dF_columns, this.dF_channels);
            this.S_dF = new Tensor(4, this.dF_num, this.dF_rows, this.dF_columns, this.dF_channels);

            // Biases and filters initialization
            // Biases are set to 0
            // Filters are set to random value from normal distribution * sqrt(2/ (num_filters * filter_rows * filter_columns))
            for (int i = 0; i < this.F_num; i++) {
                
                B.values[i] = 0.0;

                for (int j = 0; j < this.F_rows; j++) {
                    for (int k = 0; k < this.F_columns; k++) {
                        for (int l = 0; l < this.F_channels; l++) {
                            this.F.values[this.F.index(i, j, k, l)] = Utils.next_normal(Program.rand, 0, 1) * Math.Sqrt(2 / ((Double)this.F_num * this.F_rows * this.F_columns));
                        }
                    }
                }
            }
        }
      
        public Tensor forward (Tensor I) {
            
            this.I = I.pad(this.pad_size); 
            
            // Set I_samples, I_rows, and I_columns after padding
            this.I_samples = this.I.dim_1; this.I_rows = this.I.dim_2; this.I_columns = this.I.dim_3;

            this.O_samples = this.I_samples;
            this.O_rows = (this.I_rows - this.F_rows * this.dilation + this.dilation - 1) / this.stride + 1;
            this.O_columns = (this.I_columns - this.F_columns * this.dilation + this.dilation - 1) / this.stride + 1;
            this.O_channels = this.F_num;
            
            // O_2d = F_2d * I_2d + B_2d
            Tensor F_2d = Utils.F_2_mat(this.F);
            Tensor I_2d = Utils.I_to_matrix(this.I, this.F_rows, this.F_columns, this.F_channels, this.stride, this.dilation);
            Tensor B_2d = Utils.B_to_matrix(this.B, this.I_samples, this.I_rows, this.I_columns, this.F_rows, this.F_columns, this.stride, this.dilation);
            Tensor O_2d = Utils.dgemm_cs(F_2d, I_2d, B_2d);
            Tensor O = Utils.matrix_to_tensor(O_2d, O_samples, O_rows, O_columns, O_channels);
            return O;

        }
        /// <summary>
        /// Backpropagation for convolutional layer
        /// </summary>
        /// <param name="dO"> derivative of loss with respect to output </param>
        /// <returns name="dI"> if not first layer, returns derivative of loss with respect to input </returns>
        public Tensor backward(Tensor dO) {

            // Initialize ∂L/∂B and ∂L/∂F (have to store these for gradient descent)
            // Don't have to set values to 0.0 after gradient descent because a new gradient tensor is created during each backward pass
            this.dB = new Tensor(2, this.I_samples, this.dB_num);

            this.dO_rows = dO.dim_2;
            this.dO_columns = dO.dim_3;

            // CALCULATE GRADIENTS------------------------------------------------------------------------------------
                       
            // Select the I_sample from the batch
            Parallel.For(0, this.I_samples, i => {

                // ∂L/∂B
                // dB[I_sample, dB_num] is the sum of elements in dO[I_sample,__,__,dB_num]

                // Select the dB_num to be calculated
                for (int j = 0; j < this.dB_num; j++) {
                    
                    Double sum = 0.0;

                    // Loop through each elements of dO and add
                    for (int k = 0; k < this.dO_rows; k++) {
                        for (int l = 0; l < this.dO_columns; l++) {
                            sum += dO.values[dO.index(i, k, l, j)];
                        }
                    }
                    // Set the value of the bias gradient 
                    this.dB.values[i * this.dB_num + j] = sum;
                }
            });

            // Calculate ∂L/∂F
            // ∂L/∂F is the convolution of (∂L/∂O dilated by S) with stride D over I
            
            // dF_matrix = dO_matrix * I_matrix
            Tensor dO_matrix = Utils.dO_to_matrix(dO);
            // 2nd last parameter (stride of filter dO) is equal to dilation of F
            // last parameter (dilation of filter dO) is equal to stride of F
            Tensor I_matrix = Utils.I_to_matrix_backprop(this.I, dO.dim_2, dO.dim_3, F_rows, F_columns, F_channels, dilation, stride);        
            Tensor dF_matrix = new Tensor(2, F_num, F_rows * F_columns * F_channels);
            dF_matrix = Utils.dgemm_cs(dO_matrix, I_matrix, dF_matrix);
            dF = Utils.dF_matrix_to_tensor(dF_matrix, F_num, F_rows, F_columns, F_channels);

            // Calculate ∂L/∂I (if first layer, it is not needed and can return null)
            // ∂L/∂I is the full convolution of (180 rotated F dilated by D) over (∂L/∂O dilated by S and padded by (F_rows * D - D))
            if (this.needs_gradient == true) {
                
                // Size of dI is the same as size of I (before padding in the forward pass)
                this.dI_samples = this.I_samples; this.dI_rows = this.I_rows - 2 * this.pad_size; this.dI_columns = this.I_columns - 2 * this.pad_size; this.dI_channels = this.I_channels;

                // dI_matrix = F_rotated_matrix * dO_dilated_padded_matrix
                // For dO, dilate by S and pad for full convolution, then unpad by pad_size to avoid performing extra calculations, then convert to matrix
                Tensor F_rotated_matrix = Utils.F_rotated_to_matrix(this.F.rotate_180());
                Tensor dO_dilated_padded_matrix = Utils.dO_dilated_padded_to_matrix(dO.dilate(this.stride).pad(this.F_rows * this.dilation - this.dilation).unpad(this.pad_size), this.F_num, this.F_rows, this.F_columns, this.dI_samples, this.dI_rows, this.dI_columns, this.dilation);
                Tensor dI_matrix = new Tensor(2, this.dI_channels, this.dI_samples * (this.dI_rows) * (this.dI_columns));
                dI_matrix = Utils.dgemm_cs(F_rotated_matrix, dO_dilated_padded_matrix, dI_matrix);
                Tensor dI = Utils.matrix_to_tensor(dI_matrix, dI_samples, (dI_rows), (dI_columns), dI_channels);

                this.I = null;
                return dI;
            } else {
                this.I = null;
                return null;
            }
        }
    }
}
