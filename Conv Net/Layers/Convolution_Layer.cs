using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Conv_Net;
using System.Diagnostics;

namespace Conv_Net {
    class Convolution_Layer {

        private int num_input_samples;
        private int num_input_rows;
        private int num_input_columns;
        private int numInputChannels;

        private int numFilters;
        private int numFilterRows;
        private int numFilterColumns;
        private int numFilterChannels;
        
        private int numBiases;

        private int numOutputRows;
        private int numOutputColumns;
        private int numOutputChannels;

        private int stride;

        private int num_gradient_output_samples;
        private int numGradientOutputRows;
        private int numGradientOutputColumns;
        private int numGradientOutputChannels;

        private bool needsGradient;

        public Double[][,,] filters;
        public Double[][,,] biases;
        public Double[][,,] gradientFilters;
        public Double[][,,] gradientBiases;
        public Double[,,] input;

        public Tensor filter_tensor;
        public Tensor bias_tensor;
        public Tensor gradient_filter_tensor;
        public Tensor gradient_bias_tensor;
        public Tensor input_tensor;
        public int num_output_samples_tensor;

        public Convolution_Layer(int numInputChannels, int numFilters, int numFilterRows, int numFilterColumns, bool needsGradient, int stride = 1) {

            this.numInputChannels = numInputChannels;

            this.numFilters = numFilters;
            this.numFilterRows = numFilterRows;
            this.numFilterColumns = numFilterColumns;
            this.numFilterChannels = numInputChannels;
            this.stride = stride;

            this.numBiases = numFilters;

            this.needsGradient = needsGradient;

            this.filters = new Double[this.numFilters][,,];
            this.biases = new Double[this.numBiases][,,];
            this.gradientFilters = new Double[this.numFilters][,,];
            this.gradientBiases = new Double[this.numBiases][,,];


            this.bias_tensor = new Tensor(1, numFilters, 1, 1, 1);
            this.filter_tensor = new Tensor(4, numFilters, numFilterRows, numFilterColumns, numFilterChannels);
            this.gradient_bias_tensor = new Tensor(1, numFilters, 1, 1, 1);
            this.gradient_filter_tensor = new Tensor(4, numFilters, numFilterRows, numFilterColumns, numFilterChannels);


            for (int i = 0; i < this.numFilters; i++) {
                
                // Bias initialization (set to 0)
                Double[,,] tempBias = new Double[1, 1, 1];
                tempBias[0, 0, 0] = 0.0;
                this.biases[i] = tempBias;
                bias_tensor.values[i] = 0.0;

                // Filter initialization (set to random value from normal distribution * sqrt(2/ numFilters * numFilterRows * numFilterColumns))
                Double[,,] tempFilter = new Double[this.numFilterRows, this.numFilterColumns, this.numFilterChannels];
                for (int j = 0; j < this.numFilterRows; j++) {
                    for (int k = 0; k < this.numFilterColumns; k++) {
                        for (int l = 0; l < this.numFilterChannels; l++) {
                            tempFilter[j, k, l] = Program.normalDist.Sample() * Math.Sqrt(2 / ((Double)this.numFilters * this.numFilterRows * this.numFilterColumns));
                            this.filter_tensor.values[i * this.numFilterRows * this.numFilterColumns * this.numFilterChannels + j * this.numFilterColumns * this.numFilterChannels + k * this.numFilterChannels + l] = tempFilter[j, k, l];
                        }
                    }
                }
                this.filters[i] = tempFilter;

                // Initialize gradient of biases and filters with respect to loss (have to store these for gradient descent)
                Double[,,] tempBiasGradient = new Double[1, 1, 1];
                this.gradientBiases[i] = tempBiasGradient;

                Double[,,] tempFilterGradient = new Double[this.numFilterRows, this.numFilterColumns, this.numFilterChannels];
                this.gradientFilters[i] = tempFilterGradient;
            }
        }

        public Double[,,] forward (Double[,,] input) {
            this.input = input;
            this.num_input_rows = input.GetLength(0);
            this.num_input_columns = input.GetLength(1);

            Debug.Assert(this.numInputChannels == input.GetLength(2));

            this.numOutputRows= (this.num_input_rows - this.numFilterRows) / this.stride + 1;
            this.numOutputColumns = (this.num_input_columns - this.numFilterRows) / this.stride + 1;
            this.numOutputChannels = this.numFilters;
            Double[,,] output = new Double[this.numOutputRows, this.numOutputColumns, this.numOutputChannels];

            Double elementwiseProduct = 0.0;

            // Select the filter
            for (int i = 0; i < this.numFilters; i++) {

                // Select the row on the input where the top left corner of the filter is positioned 
                for (int j = 0; j <= this.num_input_rows - this.numFilterRows; j += this.stride) {

                    // Select the column on the input where the top left corner of the filter is positioned 
                    for (int k = 0; k <= this.num_input_columns - this.numFilterColumns; k += this.stride) {

                        // Loop through each element of the filter and multiply by the corresponding element of the input, add the products
                        for (int l = 0; l < this.numFilterRows; l++) {
                            for (int m = 0; m < this.numFilterColumns; m++) {
                                for (int n = 0; n < this.numFilterChannels; n++) {
                                    elementwiseProduct += filters[i][l, m, n] * input[j + l, k + m, n];
                                }
                            }
                        }
                        // Add the bias to elementwise product
                        elementwiseProduct += this.biases[i][0, 0, 0];

                        // Set the value of output
                        output[j / stride, k / stride, i] = elementwiseProduct;
                        elementwiseProduct = 0.0;
                    }
                }
            }
            return output;
        }
        public Tensor forward_tensor (Tensor input) {
            this.input_tensor = input;
            this.num_output_samples_tensor = input.dim_1;
            this.num_input_samples = input.dim_1;
            this.num_input_rows = input.dim_2;
            this.num_input_columns = input.dim_3;
            Debug.Assert(this.numInputChannels == input.dim_4);

            this.numOutputRows = (this.num_input_rows - this.numFilterRows) / this.stride + 1;
            this.numOutputColumns = (this.num_input_columns - this.numFilterRows) / this.stride + 1;
            this.numOutputChannels = this.numFilters;
            Tensor output = new Tensor(4, this.num_output_samples_tensor, this.numOutputRows, this.numOutputColumns, this.numOutputChannels);

            // Select the input image
            Parallel.For(0, this.num_output_samples_tensor, i => {
                Double elementwiseProduct = 0.0;

                // Select the filter
                for (int j = 0; j < this.numFilters; j++) {

                    // Select the row on the input where the top left corner of the filter is positioned 
                    for (int k = 0; k <= this.num_input_rows - this.numFilterRows; k += this.stride) {

                        // Select the column on the input where the top left corner of the filter is positioned 
                        for (int l = 0; l <= this.num_input_columns - this.numFilterColumns; l += this.stride) {

                            // Loop through each element of the filter and multiply by the corresponding element of the input, add the products
                            for (int m = 0; m < this.numFilterRows; m++) {
                                for (int n = 0; n < this.numFilterColumns; n++) {
                                    for (int o = 0; o < this.numFilterChannels; o++) {
                                        elementwiseProduct += this.filter_tensor.values[this.filter_tensor.index(j, m, n, o)] * input_tensor.values[this.input_tensor.index(i, (k + m), (l + n), o)];
                                    }
                                }
                            }
                            // Add the bias to elementwise product (***** check accuracy when bias != 0)
                            elementwiseProduct += this.bias_tensor.values[j];

                            // Set the value of output
                            output.values[output.index(i, (k / stride), (l / stride), j)] = elementwiseProduct;
                            elementwiseProduct = 0.0;
                        }
                    }
                }
            });
            return output;
        }

        public Double [,,] backward (Double[,,] gradientOutput) {
            this.numGradientOutputRows = gradientOutput.GetLength(0);
            this.numGradientOutputColumns = gradientOutput.GetLength(1);
            this.numGradientOutputChannels = gradientOutput.GetLength(2);

            Debug.Assert(this.numGradientOutputRows == this.numOutputRows);
            Debug.Assert(this.numGradientOutputColumns == this.numOutputColumns);
            Debug.Assert(this.numGradientOutputChannels == this.numOutputChannels);

            Double[,,] gradientInput = new Double[this.num_input_rows, this.num_input_columns, this.numInputChannels];

            Double[][,,] zeroPadded180RotatedFilters = new Double[this.numFilters][,,];

            Double elementwiseProduct = 0.0;

            // Initialize zero padded, 180 degree rotated filter
            // During backpropagation, will convolve the gradient of output over the padded, rotated filters so pad the filters on each side by (gradient of output size - 1)
            for (int i = 0; i < this.numFilters; i++) {

                // Check that the gradient of output is square
                Debug.Assert(this.numGradientOutputRows == this.numGradientOutputColumns);
                zeroPadded180RotatedFilters[i] = Utils.zeroPad(this.numGradientOutputRows - 1, Utils.rotate180(this.filters[i]));
            }

            // CALCULATING GRADIENTS-------------------------------------------------------------------------------------------------------------------------

            // Calculate gradient of loss with respect to biases
            for (int i = 0; i < this.numBiases; i++) {
                Double sum = 0.0;
                for (int j = 0; j < this.numGradientOutputRows; j++) {
                    for (int k = 0; k < this.numGradientOutputColumns; k++) {
                        sum += gradientOutput[j, k, i];
                    }
                }
                this.gradientBiases[i][0, 0, 0] += sum;
                sum = 0.0;
            }

            // Calculate gradient of loss with respect to filters
            elementwiseProduct = 0.0;

            // Select the filter gradient
            for (int i = 0; i < this.numFilters; i++) {

                // Select the Z position of the filter gradient to be calculated
                for (int j = 0; j < this.numFilterChannels; j++) {

                    // gradientFilters[gradientFilterIndex][__,__,gradientFilterPosZ] is the convolution of gradientOutput[__,__,gradientFilterIndex] over input [__,__,gradientFilterPosZ]
                    // Select the Y position of the filter gradient to be calculated (also the Y position on the input where the top left corner of the output gradient is positioned for the convolution)
                    for (int k = 0; k < this.numFilterRows; k++) {

                        // Select the X position of the filter gradient to be calculated (also the X position on the input where the top left corner of the output gradient is positioned for the convolution)
                        for (int l = 0; l < this.numFilterColumns; l++) {

                            // Loop through each element of the output gradient and multiply by the corresponding element in the input, add the products
                            for (int m = 0; m < this.numGradientOutputRows; m++) {
                                for (int n = 0; n < this.numGradientOutputColumns; n++) {
                                    elementwiseProduct += gradientOutput[m, n, i] * this.input[k + m, l + n, j];
                                }
                            }
                            // Set the value of the filter gradient
                            this.gradientFilters[i][k, l, j] += elementwiseProduct;
                            elementwiseProduct = 0.0;
                        }
                    }
                }
            }

            if (this.needsGradient == true) {
                // Calculate gradient of loss with respect to input
                elementwiseProduct = 0.0;

                // Select the Z position of the input gradient to be calculated
                for (int i = 0; i < this.numInputChannels; i++) {

                    // Loop through each filter
                    for (int j = 0; j < this.numFilters; j++) {

                        // For all filterIndex, calculate the full convolutions from right to left, bottom to top of gradientOutput[__,__, filterIndex] over 180 rotated filter [filterIndex][__,__,gradientInputPosZ] 
                        // gradientInput[__,__,gradientInputPosZ] is the elementwise sum of those convolutions

                        // Select the Y position of the input gradient to be calculated (to get the Y position on the rotated filter where the top left corner of the output gradient is positioned for convolution, reflect across X axis)
                        for (int k = 0; k < this.num_input_rows; k++) {

                            // Select the X position of the input gradient to be calculated (to get the X position on the rotated filter where the top left corner of the output gradient is positioned for convolution, reflect across Y axis)
                            for (int l = 0; l < this.num_input_columns; l++) {

                                // Loop through each element of the output gradient and multiply by the corresponding element in the filter, add the products
                                for (int m = 0; m < this.numGradientOutputRows; m++) {
                                    for (int n = 0; n < this.numGradientOutputColumns; n++) {
                                        elementwiseProduct += gradientOutput[m, n, j] * zeroPadded180RotatedFilters[j][this.num_input_rows - k - 1 + m, this.num_input_columns - l - 1 + n, i];

                                    }
                                }
                                // Increment the value of the input gradient (value is incremented each loop through filterIndex)
                                gradientInput[k, l, i] += elementwiseProduct;
                                elementwiseProduct = 0.0;
                            }
                        }
                    }
                }
                return gradientInput;
            } else {
                return null;
            }
        }

        public Tensor backward_tensor(Tensor gradient_output) {
            this.num_gradient_output_samples = gradient_output.dim_1;
            this.numGradientOutputRows = gradient_output.dim_2;
            this.numGradientOutputColumns = gradient_output.dim_3;
            this.numGradientOutputChannels = gradient_output.dim_4;

            Tensor gradient_input = new Tensor(4, this.num_input_samples, this.num_input_rows, this.num_input_columns, this.numInputChannels);
            Tensor padded_rotated_filters;
            Tensor rotated_filters;

            rotated_filters = this.filter_tensor.rotate_180();
            padded_rotated_filters = rotated_filters.zero_pad(this.numGradientOutputRows - 1);


            // Calculate Gradients

            for (int sample = 0; sample < input_tensor.dim_1; sample++) {
                
                // Bias
                for (int j = 0; j < this.numBiases; j++) {
                    Double sum = 0.0;
                    for (int k = 0; k < this.numGradientOutputRows; k++) {
                        for (int l = 0; l < this.numGradientOutputColumns; l++) {
                            sum += gradient_output.values[gradient_output.index(sample, k, l, j)];
                        }
                    }
                    this.gradient_bias_tensor.values[j] += sum;
                    sum = 0.0;
                }

                // Filter


                for (int i = 0; i < this.numFilters; i++) {
                    for (int j = 0; j < this.numFilterChannels; j++) {
                        for (int l = 0; l < this.numFilterRows; l++) {
                            for (int m = 0; m < this.numFilterColumns; m++) {
                                Double elementwise_product = 0.0;
                                for (int n = 0; n < this.numGradientOutputRows; n++) {
                                    for (int o = 0; o < this.numGradientOutputColumns; o++) {
                                        elementwise_product += gradient_output.values[gradient_output.index(sample, n, o, i)] * this.input_tensor.values[this.input_tensor.index(sample, (l + n), (m + o), j)];
                                    }
                                }
                                this.gradient_filter_tensor.values[this.gradient_filter_tensor.index(i, l, m, j)] += elementwise_product;
                                elementwise_product = 0.0;
                            }
                        }
                    }
                }
            }

                // Input
            if (this.needsGradient == true) {

                Parallel.For(0, input_tensor.dim_1, sample => {
                    for (int i = 0; i < this.numInputChannels; i++) {
                        for (int j = 0; j < this.numFilters; j++) {
                            for (int k = 0; k < this.num_input_rows; k++) {
                                for (int l = 0; l < this.num_input_columns; l++) {
                                    Double elementwise_product = 0.0;
                                    for (int m = 0; m < this.numGradientOutputRows; m++) {
                                        for (int n = 0; n < this.numGradientOutputColumns; n++) {
                                            elementwise_product += gradient_output.values[gradient_output.index(sample, m, n, j)] * padded_rotated_filters.values[padded_rotated_filters.index(j, (this.num_input_rows - k - 1 + m), (this.num_input_columns - l - 1 + n), i)];
                                        }
                                    }
                                    gradient_input.values[gradient_input.index(sample, k, l, i)] += elementwise_product;
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

        // Update filters and biases
        public void update (int batchSize) {
            for (int i=0; i < this.numFilters; i++) {
                this.biases[i][0, 0, 0] -= (this.gradientBiases[i][0, 0, 0] * Program.eta / batchSize);
                this.gradientBiases[i][0, 0, 0] = 0.0;
            
                for (int j = 0; j < this.numFilterRows; j ++) {
                    for (int k=0; k < this.numFilterColumns; k++) {
                        for (int l=0; l < this.numFilterChannels; l++) {
                            this.filters[i][j, k, l] -= (this.gradientFilters[i][j, k, l] * Program.eta / batchSize);
                            this.gradientFilters[i][j, k, l] = 0.0;
                        }
                    }
                }
            }
        }
        public void update_tensor (int batch_size) {
            for (int i = 0; i < this.numFilters; i++) {
                this.bias_tensor.values[i] -= (this.gradient_bias_tensor.values[i] * Program.eta / batch_size);
                this.gradient_bias_tensor.values[i] = 0.0;

                for (int j = 0; j < this.numFilterRows; j++) {
                    for (int k = 0; k < this.numFilterColumns; k++) {
                        for (int l = 0; l < this.numFilterChannels; l++) {
                            this.filter_tensor.values[i * (filter_tensor.dim_2 * filter_tensor.dim_3 * filter_tensor.dim_4) + j * (filter_tensor.dim_3 * filter_tensor.dim_4) + k * (filter_tensor.dim_4) + l] -= (this.gradient_filter_tensor.values[i * (gradient_filter_tensor.dim_2 * gradient_filter_tensor.dim_3 * gradient_filter_tensor.dim_4) + j * (gradient_filter_tensor.dim_3 * gradient_filter_tensor.dim_4) + k * (gradient_filter_tensor.dim_4) + l] * Program.eta / batch_size);
                            this.gradient_filter_tensor.values[i * (gradient_filter_tensor.dim_2 * gradient_filter_tensor.dim_3 * gradient_filter_tensor.dim_4) + j * (gradient_filter_tensor.dim_3 * gradient_filter_tensor.dim_4) + k * (gradient_filter_tensor.dim_4) + l] = 0.0;
                        }
                    }
                }
            }
        }
        
    }
}
