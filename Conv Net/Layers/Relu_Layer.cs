using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Relu_Layer {

        Double[,,] input;

        Tensor input_tensor;
        public Relu_Layer() {

        }

        public Double[,,] forward(Double[,,] input) {
            this.input = input;

            int numOutputRows = input.GetLength(0);
            int numOutputColumns = input.GetLength(1);
            int numOutputChannels = input.GetLength(2);
            Double[,,] output = new Double[numOutputRows, numOutputColumns, numOutputChannels];

            for (int i = 0; i < numOutputRows; i++) {
                for (int j = 0; j < numOutputColumns; j++) {
                    for (int k = 0; k < numOutputChannels; k++) {
                        output[i, j, k] = input[i, j, k] >= 0 ? input[i, j, k] : 0;
                    }
                }
            }
            return output;
        }
        public Tensor forward_tensor(Tensor input) {
            this.input_tensor = input;

            Tensor output = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);
            Parallel.For(0, output.dim_1 * output.dim_2 * output.dim_3 * output.dim_4, i => {
                output.values[i] = input.values[i] >= 0 ? input.values[i] : 0;
            });
            return output;
        }

        // Backpropagation
        // gradientOutput = dL/dO
        public Double[,,] backward(Double[,,] gradientOutput) {
            int numInputRows = this.input.GetLength(0);
            int numInputColumns = this.input.GetLength(1);
            int numInputChannels = this.input.GetLength(2);
            
            // dL/dI
            Double[,,] gradientInput = new Double[numInputRows, numInputColumns, numInputChannels];

            for (int i = 0; i < numInputRows; i++) {
                for (int j = 0; j < numInputColumns; j++) {
                    for (int k = 0; k < numInputChannels; k++) {

                        // dL/dI = dL/dO * dO/dI
                        gradientInput[i, j, k] = gradientOutput[i, j, k] * (this.input[i, j, k] >= 0 ? 1 : 0);
                    }
                }
            }
            return gradientInput;
        }

        public Tensor backward_tensor (Tensor gradientOutput) {
            Tensor gradientInput = new Tensor(this.input_tensor.dimensions, this.input_tensor.dim_1, this.input_tensor.dim_2, this.input_tensor.dim_3, this.input_tensor.dim_4);
            Parallel.For(0, gradientInput.dim_1 * gradientInput.dim_2 * gradientInput.dim_3 * gradientInput.dim_4, i => {
                gradientInput.values[i] = gradientOutput.values[i] * (this.input_tensor.values[i] >= 0 ? 1 : 0);
            });
            return gradientInput;
        }
    }
}
