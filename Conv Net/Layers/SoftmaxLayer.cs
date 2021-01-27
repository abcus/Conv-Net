using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class SoftmaxLayer {

        private Double[,,] input;

        public SoftmaxLayer() {
        }
        public Double[,,] forward(Double[,,] input) {
            this.input = input;
            int layer_size = input.GetLength(2);

            Double[,,] output = new Double[1, 1, layer_size];

            // Find max value of input array
            Double max = Double.MinValue;
            for (int i = 0; i < layer_size; i++) {
                if (input[0, 0, i] > max) {
                    max = input[0, 0, i];
                }
            }

            // Subtract max value of input array from all values
            for (int i = 0; i < layer_size; i++) {
                input[0, 0, i] -= max;
            }

            // Calculate denominator of softmax
            Double denominator = 0.0;
            for (int i = 0; i < layer_size; i++) {
                denominator += Math.Exp(input[0, 0, i]);
            }

            // Set output array
            for (int i = 0; i < layer_size; i++) {
                output[0, 0, i] = Math.Exp(input[0, 0, i]) / denominator;
            }

            return output;
        }

        public Double[,,] backward(Double[,,] inputGradient) {
            Double numerator = 0.0;
            Double denominator = 0.0;
            int layerSize = this.input.GetLength(2);
            Double[,,] output = new Double[1, 1, layerSize];

            for (int i = 0; i < layerSize; i++) {
                denominator += Math.Exp(this.input[0, 0, i]);
            }
            denominator = Math.Pow(denominator, 2);



            for (int i = 0; i < layerSize; i++) {
                numerator = 0.0;
                for (int j = 0; j < layerSize; j++) {
                    if (j != i) {
                        numerator += Math.Exp(this.input[0, 0, j]);
                    }
                }
                numerator *= Math.Exp(this.input[0, 0, i]);
                output[0, 0, i] = numerator / denominator;
            }
            return Utils.elementwiseProduct(inputGradient, output);
        }
    }
}
