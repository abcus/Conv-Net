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
            int inputSizeY = input.GetLength(0);
            int inputSizeX = input.GetLength(1);
            int inputSizeZ = input.GetLength(2);
            this.input = input;
            Double[,,] output = new Double[inputSizeY, inputSizeX, inputSizeZ];

            for (int i = 0; i < inputSizeY; i++) {
                for (int j = 0; j < inputSizeX; j++) {
                    for (int k = 0; k < inputSizeZ; k++) {
                        output[i, j, k] = input[i, j, k] >= 0 ? input[i, j, k] : 0;
                    }
                }
            }
            return output;
        }
        public Double[,,] backward(Double[,,] gradientOutput) {
            int inputSizeY = this.input.GetLength(0);
            int inputSizeX = this.input.GetLength(1);
            int inputSizeZ = this.input.GetLength(2);

            // Gradient of output with respect to input
            Double[,,] gradientLocal = new Double[inputSizeY, inputSizeX, inputSizeZ];
            
            // Gradient of loss with respect to input
            Double[,,] gradientInput = new Double[inputSizeY, inputSizeX, inputSizeZ];

            for (int i = 0; i < inputSizeY; i++) {
                for (int j = 0; j < inputSizeX; j++) {
                    for (int k = 0; k < inputSizeZ; k++) {
                        gradientLocal[i, j, k] = this.input[i, j, k] >= 0 ? 1 : 0;
                    }
                }
            }
            gradientInput = Utils.elementwiseProduct(gradientOutput, gradientLocal);
            return gradientInput;
        }
    }
}
