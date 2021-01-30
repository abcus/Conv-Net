using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class InputLayer {

        private int inputSizeY;
        private int inputSizeX;
        private int inputSizeZ;

        public InputLayer(int inputSizeY, int inputSizeX, int inputSizeZ) {
            this.inputSizeY = inputSizeY;
            this.inputSizeX = inputSizeX;
            this.inputSizeZ = inputSizeZ;
        }

        public Double[,,] forward(Double[,,] input) {
            Double[,,] output = new Double[inputSizeY, inputSizeX, inputSizeZ];
            for (int i = 0; i < inputSizeY; i ++) {
                for (int j = 0; j < inputSizeX; j ++) {
                    for (int k = 0; k < inputSizeZ; k ++) {
                        output[i, j, k] = input[i, j, k];
                    }
                }
            }
            return output;
        }
    }
}
