using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Softmax_Loss_Layer {

        public Tensor input;
        public Tensor output;
        public Tensor target;

        public Softmax_Loss_Layer() {
        }

        public Tensor forward(Tensor input) {
            this.input = input;

            Tensor output = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);

            Parallel.For(0, output.dim_1, i => {

                // Find max value of input array
                Double max = Double.MinValue;
                for (int j = 0; j < output.dim_2; j++) {
                    if (input.values[i * output.dim_2 + j] > max) {
                        max = input.values[i * output.dim_2 + j];
                    }
                }

                // Subtract max value of input array from all values
                for (int j = 0; j < output.dim_2; j++) {
                    input.values[i * output.dim_2 + j] -= max;
                }

                // Calculate denominator of softmax
                Double denominator = 0.0;
                for (int j = 0; j < output.dim_2; j++) {
                    denominator += Math.Exp(input.values[i * output.dim_2 + j]);
                }

                // Set output array
                for (int j = 0; j < output.dim_2; j++) {
                    output.values[i * output.dim_2 + j] = Math.Exp(input.values[i * output.dim_2 + j]) / denominator;
                }
            });
            this.output = output;
            return output;
        }

        /// <summary>
        /// Categorical cross entropy loss
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public Tensor loss(Tensor target) {
            this.target = target;
            Tensor loss = new Tensor(1, this.output.dim_1, 1, 1, 1);

            Parallel.For(0, loss.dim_1, i=> {
                for (int j = 0; j < this.output.dim_2; j++) {
                    loss.values[i] -= (this.target.values[i * this.output.dim_2 + j] * Math.Log(this.output.values[i * this.output.dim_2 + j]));
                }
            });
            return loss;
        }

        public Tensor backward () {

            // dL/dI
            Tensor gradient_input = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);

            Parallel.For(0, input.dim_1, i => {

                // dL/dI = (softmax output - target)
                for (int j = 0; j < input.dim_2; j++) {
                    gradient_input.values[i * input.dim_2 + j] = this.output.values[i * input.dim_2 + j] - this.target.values[i * input.dim_2 + j];
                }
            });
            return gradient_input;
        }
    }
}
