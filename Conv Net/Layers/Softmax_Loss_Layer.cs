using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Softmax_Loss_Layer {

        private int input_samples, input_rows, input_columns, input_channels;
        private int output_samples, output_rows, output_columns, output_channels;
        private int loss_samples;

        private int input_gradient_samples, input_gradient_rows, input_gradient_columns, input_gradient_channels;

        public Tensor input, output, target;

        public Softmax_Loss_Layer() {
        }

        public Tensor forward(Tensor input) {
            this.input = input;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            this.output_samples = this.input_samples;
            this.output_rows = this.input_rows;
            this.output_columns = this.input_columns;
            this.output_channels = this.input_channels;

            Tensor output = new Tensor(2, this.output_samples, this.output_rows, this.output_columns, this.output_channels);

            Parallel.For(0, this.output_samples, i => {

                // Find max value of input array
                Double max = Double.MinValue;
                for (int j = 0; j < this.output_rows; j++) {
                    if (this.input.values[i * this.output_rows+ j] > max) {
                        max = this.input.values[i * this.output_rows+ j];
                    }
                }

                // Subtract max value of input array from all values
                for (int j = 0; j < this.output_rows; j++) {
                    this.input.values[i * this.output_rows+ j] -= max;
                }

                // Calculate denominator of softmax
                Double denominator = 0.0;
                for (int j = 0; j < this.output_rows; j++) {
                    denominator += Math.Exp(this.input.values[i * this.output_rows + j]);
                }

                // Set output array
                for (int j = 0; j < this.output_rows; j++) {
                    output.values[i * this.output_rows + j] = Math.Exp(this.input.values[i * this.output_rows + j]) / denominator;
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

            this.loss_samples = this.output_samples;
            Tensor loss = new Tensor(1, this.output_samples, 1, 1, 1);

            Parallel.For(0, this.loss_samples, i=> {
                for (int j = 0; j < this.output_rows; j++) {
                    loss.values[i] -= (this.target.values[i * this.output_rows + j] * Math.Log(this.output.values[i * this.output_rows + j]));
                }
            });
            return loss;
        }

        public Tensor backward () {

            this.input_gradient_samples = this.input_samples;
            this.input_gradient_rows = this.input_rows;
            this.input_gradient_columns = this.input_columns;
            this.input_gradient_channels = this.input_channels;

            // dL/dI
            Tensor gradient_input = new Tensor(2, this.input_gradient_samples, this.input_gradient_rows, this.input_gradient_columns, this.input_gradient_channels);

            Parallel.For(0, this.input_gradient_samples, i => {

                // dL/dI = (softmax output - target)
                for (int j = 0; j < this.input_gradient_rows; j++) {
                    gradient_input.values[i * this.input_gradient_rows + j] = this.output.values[i * this.input_gradient_rows + j] - this.target.values[i * this.input_gradient_rows + j];
                }
            });
            return gradient_input;
        }
    }
}
