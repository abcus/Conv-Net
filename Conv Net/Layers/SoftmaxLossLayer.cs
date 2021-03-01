using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class SoftmaxLossLayer {

        private Double[,,] input;
        private Double[,,] softmaxOutput;
        private Double[,,] target;

        public Tensor input_tensor;
        public Tensor output_tensor;
        public Tensor target_tensor;

        public SoftmaxLossLayer() {
        
        }
        public Double[,,] forward(Double[,,] input) {
            this.input = input;
            int layerSize = input.GetLength(2);

            Double[,,] softmaxOutput = new Double[1, 1, layerSize];

            // Find max value of input array
            Double max = Double.MinValue;
            for (int i = 0; i < layerSize; i++) {
                if (input[0, 0, i] > max) {
                    max = input[0, 0, i];
                }
            }

            // Subtract max value of input array from all values
            for (int i = 0; i < layerSize; i++) {
                input[0, 0, i] -= max;
            }

            // Calculate denominator of softmax
            Double denominator = 0.0;
            for (int i = 0; i < layerSize; i++) {
                denominator += Math.Exp(input[0, 0, i]);
            }

            // Set output array
            for (int i = 0; i < layerSize; i++) {
                softmaxOutput[0, 0, i] = Math.Exp(input[0, 0, i]) / denominator;
            }
            this.softmaxOutput = softmaxOutput;
            return softmaxOutput;
        }

        public Tensor forward_tensor(Tensor input) {
            this.input_tensor = input;

            Tensor output = new Tensor(input.rank, input.num_samples, input.num_rows, input.num_columns, input.num_channels);

            Parallel.For(0, output.num_samples, i => {
                Double max = Double.MinValue;
                for (int j = 0; j < output.num_rows; j++) {
                    if (input.data[i * output.num_rows + j] > max) {
                        max = input.data[i * output.num_rows + j];
                    }
                }

                for (int j = 0; j < output.num_rows; j++) {
                    input.data[i * output.num_rows + j] -= max;
                }

                Double denominator = 0.0;
                for (int j = 0; j < output.num_rows; j++) {
                    denominator += Math.Exp(input.data[i * output.num_rows + j]);
                }

                for (int j = 0; j < output.num_rows; j++) {
                    output.data[i * output.num_rows + j] = Math.Exp(input.data[i * output.num_rows + j]) / denominator;
                }
            });
            this.output_tensor = output;
            return output;
        }

        // Categorical cross entropy loss
        public Double[,,] loss(Double[,,] target) {

            this.target = target;
            int layerSize = this.softmaxOutput.GetLength(2);
            Double[,,] loss = new Double[1, 1, 1];
            
            for (int i = 0; i < layerSize; i++) {
                loss[0, 0, 0] += (target[0, 0, i] * Math.Log(this.softmaxOutput[0, 0, i]));
            }
            loss[0, 0, 0] *= -1;
            return loss;
        }



        public Tensor loss_tensor(Tensor target) {
            this.target_tensor = target;
            Tensor loss = new Tensor(this.output_tensor.rank, this.output_tensor.num_samples, 1, 1, 1);

            Parallel.For(0, loss.num_samples, i=> {
                for (int j = 0; j < this.output_tensor.num_rows; j++) {
                    loss.data[i] -= (this.target_tensor.data[i * this.output_tensor.num_rows + j] * Math.Log(this.output_tensor.data[i * this.output_tensor.num_rows + j]));
                }
            });
            return loss;
        }

        public Double[,,] backward () {
            int layerSize = this.input.GetLength(2);

            // dL/dI
            Double[,,] gradientInput = new double[1, 1, layerSize];

            // dL/dI = (softmax output - target)
            for (int i=0; i < layerSize; i++) {
                gradientInput[0, 0, i] = this.softmaxOutput[0, 0, i] - this.target[0, 0, i];
            }
            return gradientInput;
        }

        public Tensor backward_tensor () {
            Tensor gradientInput = new Tensor(input_tensor.rank, input_tensor.num_samples, input_tensor.num_rows, input_tensor.num_columns, input_tensor.num_channels);
        
            for (int i=0; i < input_tensor.num_samples; i++) {
                for (int j=0; j < input_tensor.num_rows; j++) {
                    gradientInput.data[i * input_tensor.num_rows + j] = this.output_tensor.data[i * input_tensor.num_rows + j] - this.target_tensor.data[i * input_tensor.num_rows + j];
                }
            }
            return gradientInput;
        }
    }
}
