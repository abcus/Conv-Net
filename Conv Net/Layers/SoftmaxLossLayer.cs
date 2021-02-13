using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class SoftmaxLossLayer {

        private Double[,,] input;
        private Double[,,] softmaxOutput;
        private Double[,,] target;

        public SoftmaxLossLayer() {
        
        }
        public Double[,,] forward(Double[,,] input) {
            this.input = input;
            int layerSize = input.GetLength(2);

            Double[,,] softmaxOutput = new Double[1, 1, layerSize];

            // Find max value of input array
            Double max = Double.MinValue;
            for (int i = 0; i < layerSize; i++) {
                if (input[0, 0, i] > max) {
                    max = input[0, 0, i];
                }
            }

            // Subtract max value of input array from all values
            for (int i = 0; i < layerSize; i++) {
                input[0, 0, i] -= max;
            }

            // Calculate denominator of softmax
            Double denominator = 0.0;
            for (int i = 0; i < layerSize; i++) {
                denominator += Math.Exp(input[0, 0, i]);
            }

            // Set output array
            for (int i = 0; i < layerSize; i++) {
                softmaxOutput[0, 0, i] = Math.Exp(input[0, 0, i]) / denominator;
            }
            this.softmaxOutput = softmaxOutput;
            return softmaxOutput;
        }

        public Double[,,] categoricalCrossEntropyLoss(Double[,,] target) {

            this.target = target;
            int layerSize = this.softmaxOutput.GetLength(2);
            Double[,,] loss = new Double[1, 1, 1];
            
            for (int i = 0; i < layerSize; i++) {
                loss[0, 0, 0] += (target[0, 0, i] * Math.Log(this.softmaxOutput[0, 0, i]));
            }
            loss[0, 0, 0] *= -1;
            return loss;
        }

        public Double[,,] backward () {
            int layerSize = this.input.GetLength(2);

            // dL/dI
            Double[,,] gradientInput = new double[1, 1, layerSize];

            // dL/dI = (softmax output - target)
            for (int i=0; i < layerSize; i++) {
                gradientInput[0, 0, i] = this.softmaxOutput[0, 0, i] - this.target[0, 0, i];
            }
            return gradientInput;
        }
    }
}
