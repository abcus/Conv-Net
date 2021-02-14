﻿using System;
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
    }
}
