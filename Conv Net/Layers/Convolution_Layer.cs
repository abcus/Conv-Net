using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Conv_Net;
using System.Diagnostics;

namespace Conv_Net {
    class Convolution_Layer {

        private int input_samples, input_rows, input_columns, input_channels;
        private int num_biases;
        private int num_filters, filter_rows, filter_columns, filter_channels;
        private int output_samples, output_rows, output_columns, output_channels;

        private int input_gradient_samples, input_gradient_rows, input_gradient_columns, input_gradient_channels;
        private int num_bias_gradients;
        private int num_filter_gradients, filter_gradient_rows, filter_gradient_columns, filter_gradient_channels;
        private int gradient_output_rows, gradient_output_columns;

        private int pad_size;
        private bool needs_gradient;
        private int stride;

        public Tensor input, biases, filters;

        // Tensors to hold dL/dB and dL/dF
        // Will have separate entries for each input sample
        public Tensor gradient_biases, gradient_filters;

        public Convolution_Layer(int input_channels, int num_filters, int filter_rows, int filter_columns, int pad_size, bool needs_gradient, int stride = 1) {

            this.input_channels = input_channels;

            this.num_biases = num_filters;

            this.num_filters = num_filters;
            this.filter_rows = filter_rows;
            this.filter_columns = filter_columns;
            this.filter_channels = input_channels;

            this.num_bias_gradients = this.num_biases;

            this.num_filter_gradients = this.num_filters;
            this.filter_gradient_rows = this.filter_rows;
            this.filter_gradient_columns = this.filter_columns;
            this.filter_gradient_channels = this.filter_channels;

            this.pad_size = pad_size;
            this.needs_gradient = needs_gradient;
            this.stride = stride;

            this.biases = new Tensor(1, this.num_biases, 1, 1, 1);
            this.filters = new Tensor(4, this.num_filters, this.filter_rows, this.filter_columns, this.filter_channels);

            // Biases and filters initialization
            // Biases are set to 0
            // Filters are set to random value from normal distribution * sqrt(2/ (num_filters * filter_rows * filter_columns))
            for (int i = 0; i < this.num_filters; i++) {
                
                biases.values[i] = 0.0;

                for (int j = 0; j < this.filter_rows; j++) {
                    for (int k = 0; k < this.filter_columns; k++) {
                        for (int l = 0; l < this.filter_channels; l++) {
                            this.filters.values[this.filters.index(i, j, k, l)] = Program.normalDist.Sample() * Math.Sqrt(2 / ((Double)this.num_filters * this.filter_rows * this.filter_columns));
                        }
                    }
                }
            }
        }
        /// <summary>
        /// Feed forward for convolutional layer
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor forward (Tensor input) {
            this.input = input;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;

            this.output_samples = this.input_samples;
            this.output_rows = (this.input_rows - this.filter_rows) / this.stride + 1;
            this.output_columns = (this.input_columns - this.filter_rows) / this.stride + 1;
            this.output_channels = this.num_filters;
            Tensor output = new Tensor(4, this.output_samples, this.output_rows, this.output_columns, this.output_channels);

            // Select the input sample from the batch
            // Select the filter
            // Select the row on the input where the top left corner of the filter is positioned 
            // Select the column on the input where the top left corner of the filter is positioned 
            Parallel.For(0, this.input_samples, i => {    
                for (int j = 0; j < this.num_filters; j++) {
                    for (int k = 0; k <= this.input_rows - this.filter_rows; k += this.stride) {
                        for (int l = 0; l <= this.input_columns - this.filter_columns; l += this.stride) {

                            Double elementwise_product = 0.0;

                            // Loop through each element of the filter and multiply by the corresponding element of the input, add the products
                            for (int m = 0; m < this.filter_rows; m++) {
                                for (int n = 0; n < this.filter_columns; n++) {
                                    for (int o = 0; o < this.filter_channels; o++) {
                                        elementwise_product += this.filters.values[this.filters.index(j, m, n, o)] * this.input.values[this.input.index(i, (k + m), (l + n), o)];
                                    }
                                }
                            }
                            // Add the bias to elementwise product
                            elementwise_product += this.biases.values[j];

                            // Set the value of output
                            output.values[output.index(i, (k / stride), (l / stride), j)] = elementwise_product;
                            elementwise_product = 0.0;
                        }
                    }
                }
            });
            return output;
        }
        /// <summary>
        /// Backpropagation for convolutional layer
        /// </summary>
        /// <param name="gradient_output"></param>
        /// <returns></returns>
        public Tensor backward(Tensor gradient_output) {

            // Initialize dL/dB and dL/dF (have to store these for gradient descent)
            // Input samples is stored as the highest dimension to allow for faster access when calculating the sum across all input dimensions
            // Don't have to set values to 0.0 after updating because a new gradient tensor is created during each backward pass
            this.gradient_biases = new Tensor(2, this.num_bias_gradients, this.input_samples, 1, 1);
            this.gradient_filters = new Tensor(5, this.num_filter_gradients, this.filter_gradient_rows, this.filter_gradient_columns, this.filter_gradient_channels, this.input_samples);

            this.gradient_output_rows = gradient_output.dim_2;
            this.gradient_output_columns = gradient_output.dim_3;

            Tensor padded_rotated_filters;
            Tensor rotated_filters;

            // Create zero padded, 180 degree rotated filters
            // During backpropagation, will convolve the gradient of output over the padded, rotated filters so pad the filters on each side by (gradient output size - 1)
            rotated_filters = this.filters.rotate_180();
            padded_rotated_filters = rotated_filters.pad(this.gradient_output_rows - 1);

            // CALCULATE GRADIENTS------------------------------------------------------------------------------------

            // Select the input sample from the batch
            Parallel.For(0, this.input_samples, i => {
                // Calculate dL/dB
                // Select the bias gradient
                //      For a given input sample, gradient_biases[num_gradient_bias, input_sample] is the sum of elements in gradient_output[input_sample,__,__,num_gradient_bias]
                //      Set gradient_biases for each input sample
                for (int j = 0; j < this.num_bias_gradients; j++) {
                    Double sum = 0.0;

                    // Loop through each element of the output gradient and add
                    for (int k = 0; k < this.gradient_output_rows; k++) {
                        for (int l = 0; l < this.gradient_output_columns; l++) {
                            sum += gradient_output.values[gradient_output.index(i, k, l, j)];
                        }
                    }
                    // Set the value of the bias gradient 
                    this.gradient_biases.values[j * this.input_samples + i] = sum;
                    sum = 0.0;
                }

                // Calculate dL/dF
                // Select the filter gradient
                // Select the channel of the filter gradient to be calculated
                //      For a given input sample, gradient_filters[num_gradient_filter,__,__,gradient_filter_channel, input_sample] is the convolution of gradient_output[input_sample,__,__,num_gradient_filter] over input[input_sample,__,__,gradient_filter_channel]
                //      Increment gradient_filters for each input sample
                for (int j = 0; j < this.num_filter_gradients; j++) {
                    for (int k = 0; k < this.filter_gradient_channels; k++) {

                        // Select the row of the filter gradient to be calculated (also the row on the input where the top left corner of the output gradient is positioned for the convolution)
                        // Select the column of the filter gradient to be calculated (also the column on the input where the top left corner of the output gradient is positioned for the convolution)
                        for (int l = 0; l < this.filter_gradient_rows; l++) {
                            for (int m = 0; m < this.filter_gradient_columns; m++) {

                                Double elementwise_product = 0.0;

                                // Loop through each element of the output gradient and multiply by the corresponding element in the input, add the products
                                for (int n = 0; n < this.gradient_output_rows; n++) {
                                    for (int o = 0; o < this.gradient_output_columns; o++) {
                                        elementwise_product += gradient_output.values[gradient_output.index(i, n, o, j)] * this.input.values[this.input.index(i, (l + n), (m + o), k)];
                                    }
                                }
                                // Set the value of the filter gradient (5D tensor with input_sample as the highest dimension)
                                this.gradient_filters.values[this.gradient_filters.index(j, l, m, k, i)] = elementwise_product;
                                elementwise_product = 0.0;
                            }
                        }
                    }
                }
            });
            
            // If not first layer and dL/dI needs to be returned, calculate dL/dI
            if (this.needs_gradient == true) {
                this.input_gradient_samples = this.input_samples;
                this.input_gradient_rows = this.input_rows;
                this.input_gradient_columns = this.input_columns;
                this.input_gradient_channels = this.input_channels;
                
                Tensor gradient_input = new Tensor(4, this.input_gradient_samples, this.input_gradient_rows, this.input_gradient_columns, this.input_gradient_channels);

                // Select the input sample from the batch
                Parallel.For(0, this.input_gradient_samples, i => {

                    // Select the channel of the input gradient to be calculated
                    // Loop through each filter
                    // For all num_filters, calculate the full convolutions from right to left, bottom to top of gradientOutput[input_sample__,__,num_filter] over 180 rotated filter [num_filter,__,__,input_gradient_channel] 
                    // gradientInput[input_sample,__,__,input_gradient_channel] is the elementwise sum of those convolutions
                    for (int j = 0; j < this.input_gradient_channels; j++) {
                        for (int k = 0; k < this.num_filters; k++) {

                            // Select the row of the input gradient to be calculated (to get the row on the rotated filter where the top left corner of the output gradient is positioned for convolution, reflect across X axis)
                            // Select the column of the input gradient to be calculated (to get the column on the rotated filter where the top left corner of the output gradient is positioned for convolution, reflect across Y axis)
                            for (int l = 0; l < this.input_gradient_rows; l++) {
                                for (int m = 0; m < this.input_gradient_columns; m++) {
                                    
                                    Double elementwise_product = 0.0;

                                    // Loop through each element of the output gradient and multiply by the corresponding element in the filter, add the products  
                                    for (int n = 0; n < this.gradient_output_rows; n++) {
                                        for (int o = 0; o < this.gradient_output_columns; o++) {
                                            elementwise_product += gradient_output.values[gradient_output.index(i, n, o, k)] * padded_rotated_filters.values[padded_rotated_filters.index(k, (this.input_rows - l - 1 + n), (this.input_columns - m - 1 + o), j)];
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
                return gradient_input;
            } else {
                return null;
            }
        }

        /// <summary>
        /// Update biases and filters
        /// </summary>
        public void update () {
            Parallel.For(0, this.num_filters, i => {
                for (int s = 0; s < this.input_samples; s++) {
                    this.biases.values[i] -= (this.gradient_biases.values[i * this.input_samples + s] * Program.eta);
                }

                for (int j = 0; j < this.filter_rows; j++) {
                    for (int k = 0; k < this.filter_columns; k++) {
                        for (int l = 0; l < this.filter_channels; l++) {
                            for (int s = 0; s < this.input_samples; s++) {
                                this.filters.values[this.filters.index(i, j, k, l)] -= (this.gradient_filters.values[this.gradient_filters.index(i, j, k, l, s)] * Program.eta);
                            }
                        }
                    }
                }
            });
        }
    }
}
