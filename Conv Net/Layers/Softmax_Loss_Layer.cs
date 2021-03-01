using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Softmax_Loss_Layer {

        private Double[,,] input;
        private Double[,,] softmaxOutput;
        private Double[,,] target;

        public Tensor input_tensor;
        public Tensor output_tensor;
        public Tensor target_tensor;

        public Softmax_Loss_Layer() {
        
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

            Tensor output = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);

            Parallel.For(0, output.dim_1, i => {
                Double max = Double.MinValue;
                for (int j = 0; j < output.dim_2; j++) {
                    if (input.values[i * output.dim_2 + j] > max) {
                        max = input.values[i * output.dim_2 + j];
                    }
                }

                for (int j = 0; j < output.dim_2; j++) {
                    input.values[i * output.dim_2 + j] -= max;
                }

                Double denominator = 0.0;
                for (int j = 0; j < output.dim_2; j++) {
                    denominator += Math.Exp(input.values[i * output.dim_2 + j]);
                }

                for (int j = 0; j < output.dim_2; j++) {
                    output.values[i * output.dim_2 + j] = Math.Exp(input.values[i * output.dim_2 + j]) / denominator;
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
            Tensor loss = new Tensor(1, this.output_tensor.dim_1, 1, 1, 1);

            Parallel.For(0, loss.dim_1, i=> {
                for (int j = 0; j < this.output_tensor.dim_2; j++) {
                    loss.values[i] -= (this.target_tensor.values[i * this.output_tensor.dim_2 + j] * Math.Log(this.output_tensor.values[i * this.output_tensor.dim_2 + j]));
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
            Tensor gradientInput = new Tensor(input_tensor.dimensions, input_tensor.dim_1, input_tensor.dim_2, input_tensor.dim_3, input_tensor.dim_4);

            Parallel.For(0, input_tensor.dim_1, i => {
                for (int j = 0; j < input_tensor.dim_2; j++) {
                    gradientInput.values[i * input_tensor.dim_2 + j] = this.output_tensor.values[i * input_tensor.dim_2 + j] - this.target_tensor.values[i * input_tensor.dim_2 + j];
                }
            });
            return gradientInput;
        }
    }
}
