using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Optimizer {

        public int t; //number of updates performed, used for bias correction

        public Optimizer() {
            t = 1;
        }


        /// <summary>
        /// Stochastic gradient descent to update weights and biases of layers with trainable parameters (Fully connected, Convolution, Batch normalization)
        /// Don't need to divide by batch size because this was done in loss layer
        /// </summary>
        /// <param name="layer"></param>
        public void SGD(Base_Layer layer) {
            for (int i=0; i < layer.B.values.Length; i++) {
                layer.B.values[i] -= Program.ALPHA * layer.dB.values[i];
            }
            for (int i=0; i < layer.W.values.Length; i++) {
                layer.W.values[i] -= Program.ALPHA * layer.dW.values[i];
            }
        }

        public void Momentum(Base_Layer layer) {
            Double V_bias_correction = (1 - Math.Pow(Program.BETA_1, this.t));
            for (int i=0; i < layer.B.values.Length; i++) {
                layer.V_dB.values[i] = Program.BETA_1 * layer.V_dB.values[i] + (1 - Program.BETA_1) * layer.dB.values[i];
                layer.B.values[i] -= Program.ALPHA * layer.V_dB.values[i] / V_bias_correction;
            }
            for (int i=0; i < layer.W.values.Length; i++) {
                layer.V_dW.values[i] = Program.BETA_1 * layer.V_dW.values[i] + (1 - Program.BETA_1) * layer.dW.values[i];
                layer.W.values[i] -= Program.ALPHA * layer.V_dW.values[i] / V_bias_correction;
            }
        }

        //public void Momentum_Conv(Convolution_Layer Conv) {
        //    int F_num = Conv.dW.dim_1; int F_rows = Conv.dW.dim_2; int F_columns = Conv.dW.dim_3; int F_channels = Conv.dW.dim_4;
        //    Double V_bias_correction = (1 - Math.Pow(Program.BETA_1, this.t));

        //    for (int i = 0; i < F_num; i++) {
        //        Conv.V_dB.values[i] = Program.BETA_1 * Conv.V_dB.values[i] + (1 - Program.BETA_1) * Conv.dB.values[i];
        //        Conv.B.values[i] -= Program.ALPHA * (Conv.V_dB.values[i] / V_bias_correction);

        //        for (int j = 0; j < F_rows; j++) {
        //            for (int k = 0; k < F_columns; k++) {
        //                for (int l = 0; l < F_channels; l++) {
        //                    Conv.V_dW.values[Conv.V_dW.index(i, j, k, l)] = Program.BETA_1 * Conv.V_dW.values[Conv.V_dW.index(i, j, k, l)] + (1 - Program.BETA_1) * Conv.dW.values[Conv.dW.index(i, j, k, l)];
        //                    Conv.W.values[Conv.W.index(i, j, k, l)] -= Program.ALPHA * (Conv.V_dW.values[Conv.V_dW.index(i, j, k, l)] /V_bias_correction);
        //                }
        //            }
        //        }
        //    }
        //}

        ///// <summary>   
        ///// Update biases and filters
        ///// </summary>
        //public void ADAM_Conv (Tensor B, Tensor F, Tensor dB, Tensor dF, Tensor V_dB, Tensor S_dB, Tensor V_dF, Tensor S_dF) {
        //    int num_filters = dF.dim_1;
        //    int filter_rows = dF.dim_2;
        //    int filter_columns = dF.dim_3;
        //    int filter_channels = dF.dim_4;
        //    int input_samples = dF.dim_5;
        //    Double V_bias_correction = (1 - Math.Pow(Program.BETA_1, this.t));
        //    Double S_bias_correction = (1 - Math.Pow(Program.BETA_2, this.t));

        //    Parallel.For(0, num_filters, i => {

        //        Double dB_sum = 0;
        //        Double dF_sum = 0;

        //        for (int s = 0; s < input_samples; s++) {
        //            dB_sum += dB.values[i * input_samples + s];
        //        }
        //        V_dB.values[i] = Program.BETA_1 * V_dB.values[i] + (1 - Program.BETA_1) * dB_sum;
        //        S_dB.values[i] = Program.BETA_2 * S_dB.values[i] + (1 - Program.BETA_2) * Math.Pow(dB_sum, 2);
        //        dB_sum = 0.0;

        //        B.values[i] -= (Program.ALPHA * (V_dB.values[i] / V_bias_correction)/(Math.Sqrt(S_dB.values[i] / S_bias_correction) + Program.EPSILON));

        //        for (int j = 0; j < filter_rows; j++) {
        //            for (int k = 0; k < filter_columns; k++) {
        //                for (int l = 0; l < filter_channels; l++) {
        //                    for (int s = 0; s < input_samples; s++) {
        //                        dF_sum += dF.values[dF.index(i, j, k, l, s)];
        //                    }
        //                    V_dF.values[V_dF.index(i, j, k, l)] = Program.BETA_1 * V_dF.values[V_dF.index(i, j, k, l)] + (1 - Program.BETA_1) * dF_sum;
        //                    S_dF.values[S_dF.index(i, j, k, l)] = Program.BETA_2 * S_dF.values[S_dF.index(i, j, k, l)] + (1 - Program.BETA_2) * Math.Pow(dF_sum, 2);
        //                    dF_sum = 0.0;

        //                    F.values[F.index(i, j, k, l)] -= (Program.ALPHA * (V_dF.values[V_dF.index(i, j, k, l)]/V_bias_correction) / (Math.Sqrt(S_dF.values[S_dF.index(i, j, k, l)]/ S_bias_correction) + Program.EPSILON));
        //                }
        //            }
        //        }
        //    });
        //}



        ///// <summary>
        ///// Updates the biases and weights of the fully connected layer
        ///// Don't need to divide by batch size because this was done in softmax layer
        ///// </summary>
        //public void ADAM_FC(Tensor B, Tensor W, Tensor dB, Tensor dW, Tensor V_dB, Tensor S_dB, Tensor V_dW, Tensor S_dW) {
        //    int layer_size = dW.dim_1;
        //    int previous_layer_size = dW.dim_2;
        //    int input_samples = dW.dim_3;
        //    Double V_bias_correction = (1 - Math.Pow(Program.BETA_1, this.t));
        //    Double S_bias_correction = (1 - Math.Pow(Program.BETA_2, this.t));

        //    Parallel.For(0, layer_size, i => {

        //        Double dB_sum = 0;
        //        Double dW_sum = 0;

        //        for (int s = 0; s < input_samples; s++) {
        //            dB_sum += dB.values[i * input_samples + s];
        //        }
        //        V_dB.values[i] = Program.BETA_1 * V_dB.values[i] + (1 - Program.BETA_1) * dB_sum;
        //        S_dB.values[i] = Program.BETA_2 * S_dB.values[i] + (1 - Program.BETA_2) * Math.Pow(dB_sum, 2);
        //        dB_sum = 0.0;

        //        B.values[i] -= (Program.ALPHA * (V_dB.values[i] / V_bias_correction) / (Math.Sqrt(S_dB.values[i] / S_bias_correction) + Program.EPSILON));

        //        for (int j = 0; j < previous_layer_size; j++) {
        //            for (int s = 0; s < input_samples; s++) {
        //                dW_sum += dW.values[i * previous_layer_size * input_samples + j * input_samples + s];
        //            }
        //            V_dW.values[i * previous_layer_size + j] = Program.BETA_1 * V_dW.values[i * previous_layer_size + j] + (1 - Program.BETA_1) * dW_sum;
        //            S_dW.values[i * previous_layer_size + j] = Program.BETA_2 * S_dW.values[i * previous_layer_size + j] + (1 - Program.BETA_2) * Math.Pow(dW_sum, 2);
        //            dW_sum = 0.0;

        //            W.values[i * previous_layer_size + j] -= (Program.ALPHA * (V_dW.values[i * previous_layer_size + j] / V_bias_correction) / (Math.Sqrt(S_dW.values[i * previous_layer_size + j] / S_bias_correction) + Program.EPSILON));                    
        //        }
        //    });
        //}
    }
}
