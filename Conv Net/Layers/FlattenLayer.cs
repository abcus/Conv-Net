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
            int numInputRows = input.GetLength(0);
            int numInputColumns = input.GetLength(1);
            int numInputChannels = input.GetLength(2);
            int numOutputChannels = numInputRows * numInputColumns * numInputChannels;
            Double[,,] output = new Double[1, 1, numOutputChannels];

            for (int i = 0; i < numInputRows; i++) {
                for (int j = 0; j < numInputColumns; j++) {
                    for (int k = 0; k < numInputChannels; k++) {
                        output[0, 0, i * numInputColumns * numInputChannels + j * numInputChannels + k] = input[i, j, k];
                    }
                }
            }
            return output;
        }
    }
}
