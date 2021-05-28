using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Conv_Net {
    class Convolution_Layer {

        private int I_samples, I_rows, I_columns, I_channels;
        private int B_num;
        private int F_num, F_rows, F_columns, F_channels;
        private int O_samples, O_rows, O_columns, O_channels;

        private int dI_samples, dI_rows, dI_columns, dI_channels;
        private int dB_num;
        private int dF_num, dF_rows, dF_columns, dF_channels;
        private int dO_rows, dO_columns;

        private int pad_size;
        private bool needs_gradient;
        private int stride;
        private int dilation;

        // Input, bias, and filter tensors
        public Tensor I, B, F;

        // Tensors to hold dL/dB and dL/dF
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
                            this.F.values[this.F.index(i, j, k, l)] = Program.normalDist.Sample() * Math.Sqrt(2 / ((Double)this.F_num * this.F_rows * this.F_columns));
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
            Tensor O = new Tensor(4, this.O_samples, this.O_rows, this.O_columns, this.O_channels);

            // Select the O_sample from the batch, O_row, O_column, and O_channel (same as filter number)
            Parallel.For(0, this.O_samples, i => {    
                for (int j = 0; j < this.O_rows; j ++) {
                    for (int k = 0; k < this.O_columns; k ++) {
                        for (int l = 0; l < this.O_channels; l++) {

                            Double elementwise_product = 0.0;

                            // Loop through each element of the filter and multiply by the corresponding element of the input, add the products
                            for (int m = 0; m < this.F_rows; m++) {
                                for (int n = 0; n < this.F_columns; n++) {
                                    for (int o = 0; o < this.F_channels; o++) {
                                        elementwise_product += this.F.values[this.F.index(l, m, n, o)] * this.I.values[this.I.index(i, (j * stride + m * dilation), (k * stride + n * dilation), o)];
                                    }
                                }
                            }
                            // Add the bias to the elementwise product, set the value of output
                            O.values[O.index(i, j, k, l)] = (elementwise_product + this.B.values[l]);
                        }
                    }
                }
            });
            return O;
        }
        /// <summary>
        /// Backpropagation for convolutional layer
        /// </summary>
        /// <param name="dO"> derivative of loss with respect to output </param>
        /// <returns name="dI"> if not first layer, returns derivative of loss with respect to input </returns>
        public Tensor backward(Tensor dO) {

            // Initialize dL/dB and dL/dF (have to store these for gradient descent)
            // Don't have to set values to 0.0 after gradient descent because a new gradient tensor is created during each backward pass
            this.dB = new Tensor(2, this.I_samples, this.dB_num);
            this.dF = new Tensor(5, this.I_samples, this.dF_num, this.dF_rows, this.dF_columns, this.dF_channels);

            this.dO_rows = dO.dim_2;
            this.dO_columns = dO.dim_3;

            // CALCULATE GRADIENTS------------------------------------------------------------------------------------
                       
            // Select the I_sample from the batch
            Parallel.For(0, this.I_samples, i => {

                // dL/dB
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

                // dL/dF
                // dL/dF is the convolution of (dL/dO dilated by S) with stride D over I
                // dF[I_sample, dF_num,__,__, dF_channel] is convolution of (dO[I_sample,__,__,dF_num] dilated by S) with stride D over (I[I_sample,__,__,dF_channel])

                // Select the dF_num number, dF_row, dF_column, and dF_channel to be calculated
                for (int j = 0; j < this.dF_num; j++) {
                    for (int k = 0; k < this.dF_rows; k++) {
                        for (int l = 0; l < this.dF_columns; l++) {
                            for (int m = 0; m < this.dF_channels; m++) {
                                
                                Double dot_product = 0.0;

                                // Calculate the dot product of dO[I_sample,__,__,dF_num] and corresponding elements of I[I_sample,__,__,dF_channel]
                                for (int n = 0; n < this.dO_rows; n++) {
                                    for (int o = 0; o < this.dO_columns; o++) {
                                        dot_product += dO.values[dO.index(i, n, o, j)] * this.I.values[this.I.index(i, (k * this.dilation + n * this.stride), (l * this.dilation + o * this.stride), m)];
                                    }
                                }
                                // Set the value of the dF 
                                this.dF.values[this.dF.index(i, j, k, l, m)] = dot_product;
                            }
                        }
                    }
                }
            });

            // dL/dI (if first layer, it is not needed and can return null)
            // dL/dI is the full convolution of (180 rotated F dilated by D) over (dL/dO dilated by S and padded by [F_rows * D - D])
            // To calculate dI[I_sample,__,__, dI_channel]:
            //      For each F_num, calculate full convolution of (180 rotated F[F_num,__,__,dI_channel] dilated by D) over (dO[I_sample,__,__,F_num] dilated by S)
            //      Add all the full convolutions
           if (this.needs_gradient == true) {
                
                // Size of dI is the same as size of I (before padding in the forward pass)
                this.dI_samples = this.I_samples; this.dI_rows = this.I_rows - 2 * this.pad_size; this.dI_columns = this.I_columns - 2 * this.pad_size; this.dI_channels = this.I_channels;
                Tensor dI = new Tensor(4, this.dI_samples, this.dI_rows, this.dI_columns, this.dI_channels);

                // Dilate and pad dO, then unpad by pad_size to avoid performing extra calculations
                Tensor dO_dilated_padded = dO.dilate(this.stride).pad(this.F_rows * this.dilation - this.dilation).unpad(this.pad_size);
                Tensor F_rotated = this.F.rotate_180();
                
                // Select the sample of dI from the batch, dI_row, dI_column, and dI_channel to be calculated 
                Parallel.For(0, this.dI_samples, i => {
                    for (int j = 0; j < this.dI_rows; j++) {
                        for (int k = 0; k < this.dI_columns; k++) {
                            for (int l = 0; l < this.dI_channels; l++) {

                                // Select the F_num
                                // Calculate the dot product of 180 rotated F[F_num,__,__,dI_channel] and corresponding elements of dO[I_sample,__,__,F_num]
                                // Add the dot products across all F_num
                                Double dot_product = 0.0;
                                
                                for (int m = 0; m < this.F_num; m++) {
                                    for (int n=0; n < this.F_rows; n++) {
                                        for (int o=0; o < this.F_columns; o++) {
                                            dot_product += F_rotated.values[F_rotated.index(m, n, o, l)] * dO_dilated_padded.values[dO_dilated_padded.index(i, j + n * this.dilation, k + o * this.dilation, m)];
                                        }
                                    }
                                }
                                // Set the value of dI to be the sum of the dot products across all F_num
                                dI.values[dI.index(i, j, k, l)] = dot_product;
                            }
                        }
                    }
                }); 
                return dI;
            } else {
                return null;
            }
        }
    }
}
