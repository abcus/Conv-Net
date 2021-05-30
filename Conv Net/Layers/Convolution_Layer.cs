using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace Conv_Net {
    class Convolution_Layer {

        private int I_samples, I_rows, I_columns, I_channels;
        private int B_num;
        private int F_num, F_rows, F_columns, F_channels;
        private int O_samples, O_rows, O_columns, O_channels;

        private int dI_samples, dI_rows, dI_columns, dI_channels;
        private int dB_num;
        private int dF_num, dF_rows, dF_columns, dF_channels;
        private int dO_samples, dO_rows, dO_columns;

        public bool needs_gradient;
        public int pad_size;
        public int stride;
        public int dilation;

        public Tensor I, B, F;
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

            // Biases are initialized to 0, Filters are initializedto random value from normal distribution * sqrt(2/ (num_filters * filter_rows * filter_columns))
            for (int i = 0; i < B.values.Count(); i++) { 
                B.values[i] = 0.0; 
            }
            for (int i=0; i < F.values.Count(); i++) {
                this.F.values[i] = Utils.next_normal(Program.rand, 0, 1) * Math.Sqrt(2 / ((Double)this.F_num * this.F_rows * this.F_columns));
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
            
            // O_matrix = F_matrix * I_matrix + B_matrix
            Tensor F_matrix = Utils.F_to_matrix(this.F);
            Tensor I_matrix = Utils.I_to_matrix(this.I, this.F_rows, this.F_columns, this.F_channels, this.stride, this.dilation);
            Tensor B_matrix = Utils.B_to_matrix(this.B, this.I_samples, this.I_rows, this.I_columns, this.F_rows, this.F_columns, this.stride, this.dilation);
            Tensor O_matrix = Utils.dgemm_cs(F_matrix, I_matrix, B_matrix);
            Tensor O = Utils.matrix_to_tensor(O_matrix, O_samples, O_rows, O_columns, O_channels);
            return O;
        }
        
        public Tensor backward(Tensor dO) {

            this.dO_samples = dO.dim_1; this.dO_rows = dO.dim_2; this.dO_columns = dO.dim_3;       
            Tensor dO_matrix = Utils.dO_to_matrix(dO);

            // Calculate ∂L/∂B
            Tensor one_vector = Utils.one_vector(this.dO_samples, this.dO_rows, this.dO_columns);
            this.dB = new Tensor(1, this.dB_num);
            this.dB = Utils.dgbmv_cs(dO_matrix, one_vector, this.dB);

            // Calculate ∂L/∂F
            // ∂L/∂F is the convolution of (∂L/∂O dilated by S) with stride D over I, or dF_matrix = dO_matrix * I_matrix
            // 2nd last parameter of I_to_matrix_backprop (stride of filter dO) is equal to dilation of F
            // last parameter (dilation of filter dO) is equal to stride of F
            Tensor I_matrix = Utils.I_to_matrix_backprop(this.I, this.dO_rows, this.dO_columns, this.F_rows, this.F_columns, this.F_channels, this.dilation, this.stride);        
            Tensor dF_matrix = new Tensor(2, this.F_num, this.F_rows * this.F_columns * this.F_channels);
            dF_matrix = Utils.dgemm_cs(dO_matrix, I_matrix, dF_matrix);
            this.dF = Utils.dF_matrix_to_tensor(dF_matrix, this.F_num, this.F_rows, this.F_columns, this.F_channels);
            this.I = null;

            // Calculate ∂L/∂I (if first layer, it is not needed and can return null)
            // ∂L/∂I is the full convolution of (180 rotated F dilated by D) over (∂L/∂O dilated by S and padded by (F_rows * D - D)), or dI_matrix = F_rotated_matrix * dO_dilated_padded_matrix
            if (this.needs_gradient == true) {
                
                // Size of dI is set to the size of I (before padding in the forward pass)
                this.dI_samples = this.I_samples; this.dI_rows = this.I_rows - 2 * this.pad_size; this.dI_columns = this.I_columns - 2 * this.pad_size; this.dI_channels = this.I_channels;

                // For dO, dilate by S and pad for full convolution, then unpad by pad_size to avoid performing extra calculations, then convert to matrix
                Tensor F_rotated_matrix = Utils.F_rotated_to_matrix(this.F.rotate_180());
                Tensor dO_dilated_padded_matrix = Utils.dO_dilated_padded_to_matrix(dO.dilate(this.stride).pad(this.F_rows * this.dilation - this.dilation).unpad(this.pad_size), this.F_num, this.F_rows, this.F_columns, this.dI_samples, this.dI_rows, this.dI_columns, this.dilation);
                Tensor dI_matrix = new Tensor(2, this.dI_channels, this.dI_samples * (this.dI_rows) * (this.dI_columns));
                dI_matrix = Utils.dgemm_cs(F_rotated_matrix, dO_dilated_padded_matrix, dI_matrix);
                Tensor dI = Utils.matrix_to_tensor(dI_matrix, this.dI_samples, this.dI_rows, this.dI_columns, this.dI_channels);

                return dI;
            } else {
                return null;
            }
        }
    }
}
