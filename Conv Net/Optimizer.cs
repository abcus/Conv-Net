using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Optimizer {

        public int t; //number of updates performed, used for bias correction

        /// <summary>
        /// Optimizer to update weights and biases of layers with trainable parameters (Fully connected, Convolution, Batch normalization)
        /// Don't need to divide gradients by batch size because this was done in loss layer
        /// </summary>
        public Optimizer() {
            t = 1;
        }

        public void SGD(Base_Layer layer) {
            for (int i=0; i < layer.B.values.Length; i++) {
                layer.B.values[i] -= Program.ALPHA * layer.dB.values[i];
                layer.dB.values[i] = 0;
            }
            for (int i=0; i < layer.W.values.Length; i++) {
                layer.W.values[i] -= Program.ALPHA * layer.dW.values[i];
                layer.dW.values[i] = 0;
            }
        }

        public void Momentum(Base_Layer layer) {
            Double V_bias_correction = (1 - Math.Pow(Program.BETA_1, this.t));
            for (int i=0; i < layer.B.values.Length; i++) {
                layer.V_dB.values[i] = Program.BETA_1 * layer.V_dB.values[i] + (1 - Program.BETA_1) * layer.dB.values[i];
                layer.B.values[i] -= Program.ALPHA * layer.V_dB.values[i] / V_bias_correction;
                layer.dB.values[i] = 0;
            }
            for (int i=0; i < layer.W.values.Length; i++) {
                layer.V_dW.values[i] = Program.BETA_1 * layer.V_dW.values[i] + (1 - Program.BETA_1) * layer.dW.values[i];
                layer.W.values[i] -= Program.ALPHA * layer.V_dW.values[i] / V_bias_correction;
                layer.dW.values[i] = 0;
            }
        }

        public void ADAM(Base_Layer layer) {
            Double V_bias_correction = (1 - Math.Pow(Program.BETA_1, this.t));
            Double S_bias_correction = (1 - Math.Pow(Program.BETA_2, this.t));
            for (int i=0; i < layer.B.values.Length; i++) {
                layer.V_dB.values[i] = Program.BETA_1 * layer.V_dB.values[i] + (1 - Program.BETA_1) * layer.dB.values[i];
                layer.S_dB.values[i] = Program.BETA_2 * layer.S_dB.values[i] + (1 - Program.BETA_2) * Math.Pow(layer.dB.values[i], 2);
                layer.B.values[i] -= (Program.ALPHA * (layer.V_dB.values[i] / V_bias_correction) / (Math.Sqrt(layer.S_dB.values[i] / S_bias_correction) + Program.EPSILON));
                layer.dB.values[i] = 0;
            }
            for (int i=0; i < layer.W.values.Length; i++) {
                layer.V_dW.values[i] = Program.BETA_1 * layer.V_dW.values[i] + (1 - Program.BETA_1) * layer.dW.values[i];
                layer.S_dW.values[i] = Program.BETA_2 * layer.S_dW.values[i] + (1 - Program.BETA_2) * Math.Pow(layer.dW.values[i], 2);
                layer.W.values[i] -= (Program.ALPHA * (layer.V_dW.values[i] / V_bias_correction) / (Math.Sqrt(layer.S_dW.values[i] / S_bias_correction) + Program.EPSILON));
                layer.dW.values[i] = 0;
            }
        }

    }
}
