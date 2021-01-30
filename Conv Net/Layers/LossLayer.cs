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
            this.input = input;
            this.target = target;

            int layerSize = input.GetLength(2);
            Double[,,] output = new Double[1, 1, 1];
            Double loss = 0.0;

            for (int i = 0; i < layerSize; i++) {
                loss += (target[0, 0, i] * Math.Log(input[0, 0, i]) + (1 - target[0, 0, i]) * Math.Log(1 - input[0, 0, i]));
            }
            loss *= -1 / (Double)layerSize;
            output[0, 0, 0] = loss;
            return output;
        }

        public Double[,,] backward() {
            int layerSize = this.input.GetLength(2);
            Double[,,] gradientInput = new Double[1, 1, layerSize];
            
            for (int i = 0; i < layerSize; i++) {
                gradientInput[0, 0, i] = (-this.target[0, 0, i] + this.input[0, 0, i]) / (this.input[0, 0, i] * (1 - this.input[0, 0, i]));
            }
            return gradientInput;
        }
    }
}
