using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Conv_Net {
    class Softmax_Loss_Layer {

        private int I_samples, I_rows, I_columns, I_channels;
        private int O_samples, O_rows, O_columns, O_channels;
        private int L_samples;

        private int dI_samples, dI_rows, dI_columns, dI_channels;

        public Tensor I, O, T;

        public Softmax_Loss_Layer() {
        }

        public Tensor forward(Tensor I) {
            this.I = I; this.I_samples = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;

            this.O_samples = this.I_samples; this.O_rows = this.I_rows; this.O_columns = this.I_columns; this.O_channels = this.I_channels;

            Tensor O = new Tensor(2, this.O_samples, this.O_rows, this.O_columns, this.O_channels);

            Parallel.For(0, this.O_samples, i => {

                // Find max value of input array
                Double max = Double.MinValue;
                for (int j = 0; j < this.O_rows; j++) {
                    if (this.I.values[i * this.O_rows + j] > max) {
                        max = this.I.values[i * this.O_rows+ j];
                    }
                }

                // Subtract max value of input array from all values (to increase numerical stability in softmax)
                for (int j = 0; j < this.O_rows; j++) {
                    this.I.values[i * this.O_rows+ j] -= max;
                }

                // Calculate denominator of softmax
                Double denominator = 0.0;
                for (int j = 0; j < this.O_rows; j++) {
                    denominator += Math.Exp(this.I.values[i * this.O_rows + j]);
                }

                // Set output array
                for (int j = 0; j < this.O_rows; j++) {
                    O.values[i * this.O_rows + j] = Math.Exp(this.I.values[i * this.O_rows + j]) / denominator;
                }
            });
            this.O = O;
            return O;
        }

        /// <summary>
        /// Categorical cross entropy loss 
        /// L = - sum (target probability[class] * ln(predicted probability[class])
        /// L = - ln(predicted probability[class with target probability of 1])
        /// </summary>
        /// <param name="T"></param>
        /// <returns></returns>
        public Tensor loss(Tensor T) {
            this.T = T;

            this.L_samples = this.O_samples;
            Tensor L = new Tensor(1, this.O_samples);

            Parallel.For(0, this.L_samples, i=> {
                for (int j = 0; j < this.O_rows; j++) {
                    L.values[i] -= (this.T.values[i * this.O_rows + j] * Math.Log(this.O.values[i * this.O_rows + j]));
                }
            });
            return L;
        }
        /// <summary>
        /// Combined backpropagation for categorical cross entropy loss and softmax to increase numerical stability 
        /// </summary>
        /// <returns></returns>
        public Tensor backward () {

            this.dI_samples = this.I_samples; this.dI_rows = this.I_rows; this.dI_columns = this.I_columns; this.dI_channels = this.I_channels;

            int batch_size = this.I_samples;

            // ∂L/∂I
            Tensor dI = new Tensor(2, this.dI_samples, this.dI_rows, this.dI_columns, this.dI_channels);

            Parallel.For(0, this.dI_samples, i => {

                // ∂L/∂I = (softmax output - target)
                for (int j = 0; j < this.dI_rows; j++) {
                    dI.values[i * this.dI_rows + j] = (this.O.values[i * this.dI_rows + j] - this.T.values[i * this.dI_rows + j]) / batch_size;
                }
            });
            return dI;
        }
    }
}
