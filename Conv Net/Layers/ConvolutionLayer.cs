using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Conv_Net;

namespace Conv_Net {
    class ConvolutionLayer {

        private int numInputChannels;

        private int numFilters;
        private int numFilterRows;
        private int numFilterColumns;
        private int numFilterChannels;
        private int stride;

        private int numBiases;

        public Double[][,,] filters;
        public Double[][,,] biases;
        public Double[][,,] gradientFilters;
        public Double[][,,] gradientBiases;
        public Double[,,] input;



        public ConvolutionLayer(int numInputChannels, int numFilters, int numFilterRows, int numFilterColumns, int stride = 1) {

            this.numInputChannels = numInputChannels;

            this.numFilters = numFilters;
            this.numFilterRows = numFilterRows;
            this.numFilterColumns = numFilterColumns;
            this.numFilterChannels = numInputChannels;
            this.stride = stride;

            this.numBiases = numFilters;

            this.filters = new Double[this.numFilters][,,];
            this.biases = new Double[this.numBiases][,,];
            this.gradientFilters = new Double[this.numFilters][,,];
            this.gradientBiases = new Double[this.numBiases][,,];

            for (int i = 0; i < this.numFilters; i++) {
                
                // Bias initialization (set to 0)
                Double[,,] tempBias = new Double[1, 1, 1];
                tempBias[0, 0, 0] = 0.0;
                this.biases[i] = tempBias;

                // Filter initialization (set to random value from normal distribution * sqrt(2/ numFilters * numFilterRows * numFilterColumns))
                Double[,,] tempFilter = new Double[this.numFilterRows, this.numFilterColumns, this.numFilterChannels];
                for (int j = 0; j < this.numFilterRows; j++) {
                    for (int k = 0; k < this.numFilterColumns; k++) {
                        for (int l = 0; l < this.numFilterChannels; l++) {
                            tempFilter[j, k, l] = Program.normalDist.Sample() * Math.Sqrt(2 / ((Double)this.numFilters * this.numFilterRows * this.numFilterColumns));
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

        /*public Double[,,] forward(Double[,,] input) {
            // Dimensions of the input array
            int inputX = input.GetLength(0);
            int inputY = input.GetLength(1);
            int inputZ = input.GetLength(2);

            // Dimensions of the output arrat
            int outputX = (inputX - filterSize) / stride + 1;
            int outputY = (inputY - filterSize) / stride + 1;
            int outputZ = numFilters;

            Double[,,] output = new Double[outputX, outputY, outputZ];

            Double dotProduct = 0.0;

            for (int filter_index = 0; filter_index < numFilters; filter_index++) {
                for (int input_x_pos = 0; input_x_pos <= inputX - filterSize; input_x_pos += stride) {
                    for (int input_y_pos = 0; input_y_pos <= inputY - filterSize; input_y_pos += stride) {
                        for (int filter_x_pos = 0; filter_x_pos < filterSize; filter_x_pos++) {
                            for (int filter_y_pos = 0; filter_y_pos < filterSize; filter_y_pos++) {
                                for (int filter_z_pos = 0; filter_z_pos < inputZ; filter_z_pos++) {
                                    dotProduct += filter[filter_index][filter_x_pos, filter_y_pos, filter_z_pos] * input[input_x_pos + filter_x_pos, input_y_pos + filter_y_pos, filter_z_pos];
                                }
                            }
                        }
                        dotProduct += biases[filter_index][0, 0, 0];
                        output[input_x_pos / stride, input_y_pos / stride, filter_index] = dotProduct;
                        dotProduct = 0.0;
                    }
                }
            }
            return output;
        }*/

        /*
       // Returns gradient of cost with respect to input (dC/da = dC/dz * dz/da)
        public Double [,,] backwardInput (Double [,,] inputGradient) {
            // inputGradient dC/dz will have same dimensions as output
            // dC/da = Full convolution of dC/dz over rotated filter
            // dC/da will have same dimension as input
        }*/

        // Returns gradient of cost with respect to filter (dC/dw = dC/dz * dz/dw)
        public Double [][,,] backwardFilter (Double [,,] inputGradient, Double[,,] image, Double[][,,] filter, Double[][,,] bias) {
            // Input gradient dC/dz will have same dimension as output
            // dC/dw = convolution of dC/dz over input
            // dC/dw will have same dimension as filter



            Double[,,] dFilter = new Double[1,1,1];
            Double[,,] dBias = new Double[1, 1, 1];
            Double[,,] dImage = new Double[1,1,1];
            Double[][,,] output = new Double[3][,,];
            output[0] = dFilter;
            output[1] = dBias;
            output[2] = dImage;
            return output;
        }
    }
}
