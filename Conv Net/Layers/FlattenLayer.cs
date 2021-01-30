using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class FlattenLayer {

        public FlattenLayer() {

        }

        public Double[,,] forward(Double[,,] input) {
            int inputSizeY = input.GetLength(0);
            int inputSizeX = input.GetLength(1);
            int inputSizeZ = input.GetLength(2);
            int ouputSizeZ = inputSizeY * inputSizeX * inputSizeZ;
            Double[,,] output = new Double[1, 1, ouputSizeZ];

            for (int inputPosY = 0; inputPosY < inputSizeY; inputPosY++) {
                for (int inputPosX = 0; inputPosX < inputSizeX; inputPosX++) {
                    for (int inputPosZ = 0; inputPosZ < inputSizeZ; inputPosZ++) {
                        output[0, 0, inputPosY * inputSizeX * inputSizeZ + inputPosX * inputSizeZ + inputPosZ] = input[inputPosY, inputPosX, inputPosZ];
                    }
                }
            }
            return output;
        }
    }
}
