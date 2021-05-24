using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Conv_Net;
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
            
            this.F_num = F_num;
            this.F_rows = (F_rows - 1) * dilation + 1;
            this.F_columns = (F_columns - 1) * dilation + 1;
            this.F_channels = I_channels;

            this.dB_num = this.B_num;
            
            this.dF_num = this.F_num; 
            this.dF_rows = this.F_rows; 
            this.dF_columns = this.F_columns; 
            this.dF_channels = this.F_channels;

            this.pad_size = pad_size;
            this.needs_gradient = needs_gradient;
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
        /// <summary>
        /// Feed forward for convolutional layer
        /// </summary>
        /// <returns></returns>
        public Tensor forward (Tensor I) {
            this.I = I;
            if (this.pad_size != 0) { this.I = I.pad(this.pad_size); }
            this.I_samples = this.I.dim_1;
            this.I_rows = this.I.dim_2;
            this.I_columns = this.I.dim_3;

            this.O_samples = this.I_samples;
            this.O_rows = (this.I_rows - this.F_rows) / this.stride + 1;
            this.O_columns = (this.I_columns - this.F_columns) / this.stride + 1;
            this.O_channels = this.F_num;
            Tensor O = new Tensor(4, this.O_samples, this.O_rows, this.O_columns, this.O_channels);

            // Select the input sample from the batch
            // Select the output row
            // Select the output column
            // Select the output channel (or filter number)
            Parallel.For(0, this.I_samples, i => {    
                for (int j = 0; j < this.O_rows; j ++) {
                    for (int k = 0; k < this.O_columns; k ++) {
                        for (int l = 0; l < this.O_channels; l++) {

                            Double elementwise_product = 0.0;

                            // Loop through each element of the filter and multiply by the corresponding element of the input, add the products
                            for (int m = 0; m < this.F_rows; m++) {
                                for (int n = 0; n < this.F_columns; n++) {
                                    for (int o = 0; o < this.F_channels; o++) {
                                        elementwise_product += this.F.values[this.F.index(l, m, n, o)] * this.I.values[this.I.index(i, (j * stride + m), (k * stride + n), o)];
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
        /// <param name="dO"></param>
        /// <returns></returns>
        public Tensor backward(Tensor dO) {

            // Initialize dL/dB and dL/dF (have to store these for gradient descent)
            // Input samples is stored as the highest dimension to allow for faster access when calculating the sum across all input dimensions
            // Don't have to set values to 0.0 after updating because a new gradient tensor is created during each backward pass
            this.dB = new Tensor(2, this.I_samples, this.dB_num);
            this.dF = new Tensor(5, this.I_samples, this.dF_num, this.dF_rows, this.dF_columns, this.dF_channels);

            this.dO_rows = dO.dim_2;
            this.dO_columns = dO.dim_3;

            Tensor rotated_F;
            Tensor padded_rotated_F;
            
            // Create zero padded, 180 degree rotated filters
            // During backpropagation, will convolve the gradient of output over the padded, rotated filters so pad the filters on each side by (gradient output size - 1)
            rotated_F = this.F.rotate_180();
            padded_rotated_F = rotated_F.pad(this.dO_rows - 1);

            // CALCULATE GRADIENTS------------------------------------------------------------------------------------

            // Select the input sample from the batch
            Parallel.For(0, this.I_samples, i => {
                // Calculate dL/dB
                // Select the bias gradient
                //      For a given input sample, gradient_biases[num_gradient_bias, input_sample] is the sum of elements in gradient_output[input_sample,__,__,num_gradient_bias]
                //      Set gradient_biases for each input sample
                for (int j = 0; j < this.dB_num; j++) {
                    Double sum = 0.0;

                    // Loop through each element of the output gradient and add
                    for (int k = 0; k < this.dO_rows; k++) {
                        for (int l = 0; l < this.dO_columns; l++) {
                            sum += dO.values[dO.index(i, k, l, j)];
                        }
                    }
                    // Set the value of the bias gradient 
                    this.dB.values[i * this.dB_num + j] = sum;
                    sum = 0.0;
                }

                // Calculate dL/dF
                // Select the filter gradient
                // Select the channel of the filter gradient to be calculated
                //      For a given input sample, gradient_filters[num_gradient_filter,__,__,gradient_filter_channel, input_sample] is the convolution of gradient_output[input_sample,__,__,num_gradient_filter] over input[input_sample,__,__,gradient_filter_channel]
                //      Increment gradient_filters for each input sample
                for (int j = 0; j < this.dF_num; j++) {
                    for (int k = 0; k < this.dF_channels; k++) {

                        // Select the row of the filter gradient to be calculated (also the row on the input where the top left corner of the output gradient is positioned for the convolution)
                        // Select the column of the filter gradient to be calculated (also the column on the input where the top left corner of the output gradient is positioned for the convolution)
                        for (int l = 0; l < this.dF_rows; l++) {
                            for (int m = 0; m < this.dF_columns; m++) {

                                Double elementwise_product = 0.0;

                                // Loop through each element of the output gradient and multiply by the corresponding element in the input, add the products
                                for (int n = 0; n < this.dO_rows; n++) {
                                    for (int o = 0; o < this.dO_columns; o++) {
                                        elementwise_product += dO.values[dO.index(i, n, o, j)] * this.I.values[this.I.index(i, (l + n), (m + o), k)];
                                    }
                                }
                                // Set the value of the filter gradient (5D tensor with input_sample as the highest dimension)
                                this.dF.values[this.dF.index(i, j, l, m, k)] = elementwise_product;
                                elementwise_product = 0.0;
                            }
                        }
                    }
                }
            });
            
            // If not first layer and dL/dI needs to be returned, calculate dL/dI
            if (this.needs_gradient == true) {
                this.dI_samples = this.I_samples;
                this.dI_rows = this.I_rows;
                this.dI_columns = this.I_columns;
                this.dI_channels = this.I_channels;
                
                Tensor gradient_input = new Tensor(4, this.dI_samples, this.dI_rows, this.dI_columns, this.dI_channels);

                // Select the input sample from the batch
                Parallel.For(0, this.dI_samples, i => {

                    // Select the channel of the input gradient to be calculated
                    // Loop through each filter
                    // For all num_filters, calculate the full convolutions from right to left, bottom to top of gradientOutput[input_sample__,__,num_filter] over 180 rotated filter [num_filter,__,__,input_gradient_channel] 
                    // gradientInput[input_sample,__,__,input_gradient_channel] is the elementwise sum of those convolutions
                    for (int j = 0; j < this.dI_channels; j++) {
                        for (int k = 0; k < this.F_num; k++) {

                            // Select the row of the input gradient to be calculated (to get the row on the rotated filter where the top left corner of the output gradient is positioned for convolution, reflect across X axis)
                            // Select the column of the input gradient to be calculated (to get the column on the rotated filter where the top left corner of the output gradient is positioned for convolution, reflect across Y axis)
                            for (int l = 0; l < this.dI_rows; l++) {
                                for (int m = 0; m < this.dI_columns; m++) {
                                    
                                    Double elementwise_product = 0.0;

                                    // Loop through each element of the output gradient and multiply by the corresponding element in the filter, add the products  
                                    for (int n = 0; n < this.dO_rows; n++) {
                                        for (int o = 0; o < this.dO_columns; o++) {
                                            elementwise_product += dO.values[dO.index(i, n, o, k)] * padded_rotated_F.values[padded_rotated_F.index(k, (this.I_rows - l - 1 + n), (this.I_columns - m - 1 + o), j)];
                                        }
                                    }
                                    // Increment the value of the input gradient (value is incremented each loop through num_filters)
                                    gradient_input.values[gradient_input.index(i, l, m, j)] += elementwise_product;
                                    elementwise_product = 0.0;
                                }
                            }
                        }
                    }
                }); 
                return gradient_input.unpad(this.pad_size);
            } else {
                return null;
            }
        }
    }
}
