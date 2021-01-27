using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class FullyConnectedLayer {

        private int previous_layer_size;
        private int layer_size;
        public Double[][,,] weights;
        public Double[][,,] biases;
        private Double[][,,] weightGradients;
        private Double[][,,] biasGradients;
        public Double[,,] inputs;

        public FullyConnectedLayer(int previous_layer_size, int layer_size) {
            this.previous_layer_size = previous_layer_size;
            this.layer_size = layer_size;
            this.weights = new Double[layer_size][,,];
            this.biases = new Double[layer_size][,,];
            this.weightGradients = new Double[layer_size][,,];
            this.biasGradients = new Double[layer_size][,,];

            for (int i = 0; i < layer_size; i++) {
                Double[,,] temp_weights = new Double[1, 1, previous_layer_size];

                for (int j = 0; j < previous_layer_size; j++) {
                    temp_weights[0, 0, j] = Program.normalDist.Sample() * Math.Sqrt(2 / (Double)previous_layer_size);
                }
                this.weights[i] = temp_weights;
            }

            for (int i = 0; i < layer_size; i++) {
                Double[,,] temp_biases = new Double[1, 1, 1];
                temp_biases[0, 0, 0] = 0.0;
                this.biases[i] = temp_biases;
            }

            for (int i = 0; i < layer_size; i++) {
                Double[,,] tempWeightGradient = new Double[1, 1, previous_layer_size];
                this.weightGradients[i] = tempWeightGradient;

                Double[,,] tempBiasGradient = new Double[1, 1, 1];
                this.biasGradients[i] = tempBiasGradient;
            }
        }

        public Double[,,] forward(Double[,,] input) {
            this.inputs = input;
            Double[,,] output = new Double[1, 1, layer_size];
            for (int i = 0; i < layer_size; i++) {

                output[0, 0, i] = Utils.dotProduct(input, weights[i]) + biases[i][0, 0, 0];
            }
            return output;
        }

        public Double[,,] backward(Double[,,] inputGradient) {
            Double[,,] output = new Double[1, 1, previous_layer_size];
            for (int i = 0; i < previous_layer_size; i++) {
                output[0, 0, i] = Utils.dotProduct(inputGradient, Utils.transpose(this.weights)[i]);
            }
            return output;
        }
    }
}
