using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class ReluLayer {

        Double[,,] input;

        public ReluLayer() {

        }

        public Double[,,] forward(Double[,,] input) {
            int input_size_x = input.GetLength(0);
            int input_size_y = input.GetLength(1);
            int input_size_z = input.GetLength(2);
            this.input = input;
            Double[,,] output = new Double[input_size_x, input_size_y, input_size_z];

            for (int i = 0; i < input_size_x; i++) {
                for (int j = 0; j < input_size_y; j++) {
                    for (int k = 0; k < input_size_z; k++) {
                        output[i, j, k] = input[i, j, k] >= 0 ? input[i, j, k] : 0;
                    }
                }
            }
            return output;
        }

        public Double[,,] backward(Double[,,] inputGradient) {
            int input_size_x = this.input.GetLength(0);
            int input_size_y = this.input.GetLength(1);
            int input_size_z = this.input.GetLength(2);
            Double[,,] output = new Double[input_size_x, input_size_y, input_size_z];

            for (int i = 0; i < input_size_x; i++) {
                for (int j = 0; j < input_size_y; j++) {
                    for (int k = 0; k < input_size_z; k++) {
                        output[i, j, k] = this.input[i, j, k] >= 0 ? 1 : 0;
                    }
                }
            }
            return Utils.elementwiseProduct(inputGradient, output);
        }
    }
}
