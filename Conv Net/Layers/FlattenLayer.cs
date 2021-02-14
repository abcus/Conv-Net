﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class FlattenLayer {

        private int numInputRows;
        private int numInputColumns;
        private int numInputChannels;
        private int numOutputChannels;

        public FlattenLayer() {

        }

        public Double[,,] forward(Double[,,] input) {
            this.numInputRows = input.GetLength(0);
            this.numInputColumns = input.GetLength(1);
            this.numInputChannels = input.GetLength(2);
            this.numOutputChannels = this.numInputRows * this.numInputColumns * this.numInputChannels;
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


        public Double[,,] backward(Double[,,] gradientOutput) {
            Double[,,] gradientInput = new Double[this.numInputRows, this.numInputColumns, this.numInputChannels];

            for (int i = 0; i < numInputRows; i++) {
                for (int j = 0; j < numInputColumns; j++) {
                    for (int k = 0; k < numInputChannels; k++) {
                        gradientInput[i, j, k] = gradientOutput[0, 0, i * this.numInputColumns * this.numInputChannels + j * this.numInputChannels + k];
                    }
                }
            }
            return gradientInput;
        }
    }
}
