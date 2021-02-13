using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class InputLayer {

        private int numInputRows;
        private int numInputColumns;
        private int numInputChannels;

        public InputLayer(int numInputRows, int numInputColumns, int numInputChannels) {
            this.numInputRows = numInputRows;
            this.numInputColumns = numInputColumns;
            this.numInputChannels = numInputChannels;
        }

        public Double[,,] forward(Double[,,] input) {
            Double[,,] output = new Double[numInputRows, numInputColumns, numInputChannels];
            for (int i = 0; i < numInputRows; i ++) {
                for (int j = 0; j < numInputColumns; j ++) {
                    for (int k = 0; k < numInputChannels; k ++) {
                        output[i, j, k] = input[i, j, k];
                    }
                }
            }
            return output;
        }
    }
}
