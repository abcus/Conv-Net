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
        public void SGD_Conv (Tensor B, Tensor F, Tensor dB, Tensor dF, Tensor V_dB, Tensor S_dB, Tensor V_dF, Tensor S_dF) {

            int num_filters = dF.dim_1;
            int filter_rows = dF.dim_2;
            int filter_columns = dF.dim_3;
            int filter_channels = dF.dim_4;
            int input_samples = dF.dim_5;

            Parallel.For(0, num_filters, i => {

                Double dB_sum = 0;
                Double dF_sum = 0;

                for (int s = 0; s < input_samples; s++) {
                    dB_sum += dB.values[i * input_samples + s];
                }
                B.values[i] -= (Program.ALPHA * dB_sum);
                dB_sum = 0;

                for (int j = 0; j < filter_rows; j++) {
                    for (int k = 0; k < filter_columns; k++) {
                        for (int l = 0; l < filter_channels; l++) {
                            for (int s = 0; s < input_samples; s++) {
                                dF_sum += dF.values[dF.index(i, j, k, l, s)];
                            }
                            F.values[F.index(i, j, k, l)] -= (Program.ALPHA * dF_sum);
                            dF_sum = 0;
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
