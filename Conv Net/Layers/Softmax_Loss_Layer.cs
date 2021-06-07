using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Conv_Net {
    class Softmax_Loss_Layer : Base_Layer {

        private int I_samples, I_rows, I_columns, I_channels;
        private Tensor O, T;

        public Softmax_Loss_Layer() {
        }

        public override Tensor forward(Tensor I) {
            this.I_samples = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;

            Parallel.For(0, this.I_samples, i => {

                // Find max value of input array
                Double max = Double.MinValue;
                for (int j = 0; j < this.I_rows; j++) {
                    if (I.values[i * this.I_rows + j] > max) {
                        max = I.values[i * this.I_rows+ j];
                    }
                }

                // Subtract max value of input array from all values (to increase numerical stability in softmax)
                for (int j = 0; j < this.I_rows; j++) {
                    I.values[i * this.I_rows+ j] -= max;
                }

                // Calculate denominator of softmax
                Double denominator = 0.0;
                for (int j = 0; j < this.I_rows; j++) {
                    denominator += Math.Exp(I.values[i * this.I_rows + j]);
                }

                // Set output array
                for (int j = 0; j < this.I_rows; j++) {
                    I.values[i * this.I_rows + j] = Math.Exp(I.values[i * this.I_rows + j]) / denominator;
                }
            });
            // O is calculated in-place from I
            this.O = I;
            return I;
        }

        /// <summary>
        /// Categorical cross entropy loss 
        /// L = - sum (target probability[class] * ln(predicted probability[class])
        /// L = - ln(predicted probability[class with target probability of 1])
        /// </summary>
        /// <param name="T"></param>
        /// <returns name="L"> Scalar </returns>
        public override Tensor loss(Tensor T) {
            this.T = T;
            Tensor L = new Tensor(1, 1);

            Double difference = 0.0;
            for (int i=0; i < O.values.Length; i++) {
                difference -= (this.T.values[i] * Math.Log(this.O.values[i]));
            }
            L.values[0] = difference / this.I_samples;
            return L;
        }
        /// <summary>
        /// Combined backpropagation for categorical cross entropy loss and softmax to increase numerical stability 
        /// </summary>
        /// <returns></returns>
        public override Tensor backward (Tensor dO) {

            int batch_size = this.I_samples;

            // ∂L/∂I
            Tensor dI = new Tensor(2, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            Parallel.For(0, this.I_samples, i => {

                // ∂L/∂I = (softmax output - target)
                for (int j = 0; j < this.I_rows; j++) {
                    dI.values[i * this.I_rows + j] = (this.O.values[i * this.I_rows + j] - this.T.values[i * this.I_rows + j]) / batch_size;
                }
            });
            this.O = null;
            this.T = null;
            return dI;
        }
    }
}
