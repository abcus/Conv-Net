using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Conv_Net;

namespace Conv_Net {
    class ConvolutionLayer {

        private int numFilters;
        private int filterSize;
        private int stride;
        private Double[][,,] filter;
        private Double[][,,] biases;


        public ConvolutionLayer(int inputZ, int numFilters, int filterSize, int stride = 1) {

            this.numFilters = numFilters;
            this.filterSize = filterSize;
            this.stride = stride;
            filter = new Double[numFilters][,,];
            biases = new Double[numFilters][,,];

            for (int i=0; i < numFilters; i++) {
                filter[i] = new Double[filterSize, filterSize, inputZ];
                biases[i] = new Double[1, 1, 1];
            }
            // Initialize filter weights
        }

        public Double[,,] forward(Double[,,] input) {
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
        }

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
