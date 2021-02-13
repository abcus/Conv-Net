using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class LossLayer {

        private Double[,,] input;
        private Double[,,] target;

        public LossLayer() {

        }

        public Double[,,] forward(Double[,,] input, Double[,,] target) {
            
            // this.input is output of NN
            this.input = input;
            this.target = target;

            int layerSize = input.GetLength(2);
            Double[,,] output = new Double[1, 1, 1];
            Double loss = 0.0;

            for (int i = 0; i < layerSize; i++) {
                output[0, 0, 0] += (target[0, 0, i] * Math.Log(input[0, 0, i]));
            }
            output[0, 0, 0] *= -1;
            return output;
        }

        public Double[,,] backward(Double [,,] gradientOutput) {
            int layerSize = this.input.GetLength(2);
            
            // dL/dI
            Double[,,] gradientInput = new Double[1, 1, layerSize];
            
            for (int i = 0; i < layerSize; i++) {
                
                // dL/dI = dL/dO * dO/dI = dL/dL * dL/dI = 1 * dL/dI
                gradientInput[0, 0, i] = gradientOutput[0, 0, 0] * ((-this.target[0, 0, i] + this.input[0, 0, i]) / (this.input[0, 0, i] * (1 - this.input[0, 0, i])));
            }
            return gradientInput;
        }
    }
}
