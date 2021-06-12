﻿using System;
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
        public int O_samples, O_rows, O_columns, O_channels;

        public int dI_samples, dI_rows, dI_columns, dI_channels;
        public int dB_num;
        public int dW_num, dW_rows, dW_columns, dW_channels;
        public int dO_samples, dO_rows, dO_columns;

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

            this.dB_num = this.B_num;
            this.dW_num = this.W_num; this.dW_rows = this.W_rows; this.dW_columns = this.W_columns; this.dW_channels = this.W_channels;

            this.needs_gradient = needs_gradient;
            this.pad_size = pad_size;
            this.stride = stride;
            this.dilation = dilation;
            this.groups = groups;

            this.B = new Tensor(2, this.B_num, 1);
            this.V_dB = new Tensor(2, this.dB_num, 1);
            this.S_dB = new Tensor(2, this.dB_num, 1);

            this.W = new Tensor(4, this.W_num, this.W_rows, this.W_columns, this.W_channels);
            this.V_dW = new Tensor(4, this.dW_num, this.dW_rows, this.dW_columns, this.dW_channels);
            this.S_dW = new Tensor(4, this.dW_num, this.dW_rows, this.dW_columns, this.dW_channels);

            // Biases are initialized to 0, Filters are initializedto random value from normal distribution * sqrt(2/ (num_filters * filter_rows * filter_columns))
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
            this.I_samples = this.I.dim_1; this.I_rows = this.I.dim_2; this.I_columns = this.I.dim_3;
            this.O_samples = this.I_samples;
            this.O_rows = (this.I_rows - this.W_rows * this.dilation + this.dilation - 1) / this.stride + 1;
            this.O_columns = (this.I_columns - this.W_columns * this.dilation + this.dilation - 1) / this.stride + 1;
            this.O_channels = this.W_num;

            // O_matrix = F_matrix * I_matrix + B_matrix
            Tensor O_matrix = Utils.forward_Conv_CPU(this);
            Tensor O = Utils.matrix_to_tensor(O_matrix, O_samples, O_rows, O_columns, O_channels);
            return O;
        }
        
        public override Tensor backward(Tensor dO) {

            this.dO_samples = dO.dim_1; this.dO_rows = dO.dim_2; this.dO_columns = dO.dim_3;       
            Tensor dO_matrix = Utils.dO_to_matrix(dO);

            // Calculate ∂L/∂B
            Tensor column_vector_1 = Utils.column_vector_1(this.dO_samples * this.dO_rows * this.dO_columns);
            this.dB = new Tensor(2, this.dB_num, 1);
            this.dB = Utils.dgemm_cs(dO_matrix, column_vector_1, this.dB);

            // Calculate ∂L/∂F
            // ∂L/∂F is the convolution of (∂L/∂O dilated by S) with stride D over I, or dF_matrix = dO_matrix * I_matrix
            // 2nd last parameter of I_to_matrix_backprop (stride of filter dO) is equal to dilation of F
            // last parameter (dilation of filter dO) is equal to stride of F
            Tensor I_matrix = Utils.I_to_matrix_backprop(this.I, this.dO_rows, this.dO_columns, this.W_rows, this.W_columns, this.W_channels, this.dilation, this.stride);        
            Tensor dF_matrix = new Tensor(2, this.W_num, this.W_rows * this.W_columns * this.W_channels);
            dF_matrix = Utils.dgemm_cs(dO_matrix, I_matrix, dF_matrix);
            this.dW = Utils.dF_matrix_to_tensor(dF_matrix, this.W_num, this.W_rows, this.W_columns, this.W_channels);
            this.I = null;

            // Calculate ∂L/∂I (if first layer, it is not needed and can return null)
            // ∂L/∂I is the full convolution of (180 rotated F dilated by D) over (∂L/∂O dilated by S and padded by (F_rows * D - D)), or dI_matrix = F_rotated_matrix * dO_dilated_padded_matrix
            if (this.needs_gradient == true) {
                
                // Size of dI is set to the size of I (before padding in the forward pass)
                this.dI_samples = this.I_samples; this.dI_rows = this.I_rows - 2 * this.pad_size; this.dI_columns = this.I_columns - 2 * this.pad_size; this.dI_channels = this.I_channels;

                // For dO, dilate by S and pad for full convolution, then unpad by pad_size to avoid performing extra calculations, then convert to matrix
                Tensor F_rotated_matrix = Utils.F_rotated_to_matrix(this.W.rotate_180());
                Tensor dO_dilated_padded_matrix = Utils.dO_dilated_padded_to_matrix(dO.dilate(this.stride).pad(this.W_rows * this.dilation - this.dilation).unpad(this.pad_size), this.W_num, this.W_rows, this.W_columns, this.dI_samples, this.dI_rows, this.dI_columns, this.dilation);
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
