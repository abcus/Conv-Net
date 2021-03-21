using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Gradient_Descent {

        public int t;

        public Gradient_Descent() {
            t = 0;
        }

        /// <summary>
        /// Update biases and filters
        /// </summary>
        public void SGD_Conv (Tensor biases, Tensor filters, Tensor gradient_biases, Tensor gradient_filters) {

            int num_filters = gradient_filters.dim_1;
            int filter_rows = gradient_filters.dim_2;
            int filter_columns = gradient_filters.dim_3;
            int filter_channels = gradient_filters.dim_4;
            int input_samples = gradient_filters.dim_5;

            Parallel.For(0, num_filters, i => {
                for (int s = 0; s < input_samples; s++) {
                    biases.values[i] -= (gradient_biases.values[i * input_samples + s] * Program.ALPHA);
                }

                for (int j = 0; j < filter_rows; j++) {
                    for (int k = 0; k < filter_columns; k++) {
                        for (int l = 0; l < filter_channels; l++) {
                            for (int s = 0; s < input_samples; s++) {
                                filters.values[filters.index(i, j, k, l)] -= (gradient_filters.values[gradient_filters.index(i, j, k, l, s)] * Program.ALPHA);
                            }
                        }
                    }
                }
            });
        }

        /// <summary>
        /// Updates the biases and weights of the fully connected layer
        /// Don't need to divide by batch size because this was done in softmax layer
        /// </summary>
        public void SGD_FC(Tensor biases, Tensor weights, Tensor gradient_biases, Tensor gradient_weights) {
            int layer_size = gradient_weights.dim_1;
            int previous_layer_size = gradient_weights.dim_2;
            int input_samples = gradient_weights.dim_3;

            Parallel.For(0, layer_size, i => {
                for (int s = 0; s < input_samples; s++) {
                    biases.values[i] -= (gradient_biases.values[i * input_samples + s] * Program.ALPHA);
                }
                for (int j = 0; j < previous_layer_size; j++) {
                    for (int s = 0; s < input_samples; s++) {
                        weights.values[i * previous_layer_size + j] -= (gradient_weights.values[i * previous_layer_size * input_samples + j * input_samples + s] * Program.ALPHA);
                    }
                }
            });
        }
    }
}
