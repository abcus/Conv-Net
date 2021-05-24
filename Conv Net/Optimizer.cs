﻿using System;
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
        /// Updates the biases and weights of the fully connected layer
        /// Don't need to divide by batch size because this was done in softmax layer
        /// </summary>
        public void SGD_FC(Fully_Connected_Layer FC) {
            int I_samples = FC.dW.dim_1;
            int layer_size = FC.dW.dim_2;
            int previous_layer_size = FC.dW.dim_3;

            for (int i = 0; i < I_samples; i++) {
                for (int j = 0; j < layer_size; j++) {    
                    
                    FC.B.values[j] -= (FC.dB.values[i * layer_size + j] * Program.ALPHA);

                    for (int k = 0; k < previous_layer_size; k++) {
                        FC.W.values[j * previous_layer_size + k] -= (FC.dW.values[i * layer_size * previous_layer_size + j * previous_layer_size + k] * Program.ALPHA);
                    }
                }
            }
        }

        public void SGD_Conv(Convolution_Layer Conv) {
            int I_samples = Conv.dF.dim_1;
            int F_num = Conv.dF.dim_2;
            int F_rows = Conv.dF.dim_3;
            int F_columns = Conv.dF.dim_4;
            int F_channels = Conv.dF.dim_5;

            for (int i = 0; i < I_samples; i++) {
                for (int j = 0; j < F_num; j++) {
                    
                    Conv.B.values[j] -= Conv.dB.values[i * F_num + j] * Program.ALPHA;

                    for (int k = 0; k < F_rows; k++) {
                        for (int l = 0; l < F_columns; l++) {
                            for (int m = 0; m < F_channels; m++) {
                                Conv.F.values[Conv.F.index(j, k, l, m)] -= Conv.dF.values[Conv.dF.index(i, j, k, l, m)] * Program.ALPHA;
                            }
                        }
                    }
                }
            }
        }

        

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