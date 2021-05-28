using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Dropout_Layer {

        // Probability that inputs are set to 0
        private Double p;
        private Double scaling_factor;

        private int I_dimensions, I_samples, I_rows, I_columns, I_channels;
        private int output_samples, output_rows, output_columns, output_channels;

        // dO/dI
        private Tensor gradient_local;

        public Dropout_Layer(Double p) {
            this.p = p;
            this.scaling_factor = 1 / (1 - this.p);
        }

        public Tensor forward(Tensor input) {
            
            this.I_dimensions = input.dimensions;
            this.I_samples = input.dim_1;
            this.I_rows = input.dim_2;
            this.I_columns = input.dim_3;
            this.I_channels = input.dim_4;

            this.gradient_local = new Tensor(this.I_dimensions, this.I_samples, this.I_rows, this.I_columns, this.I_channels);
            Tensor output = new Tensor(this.I_dimensions, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            for (int i=0; i < this.I_samples * this.I_rows * this.I_columns * this.I_channels; i++) {
                if (Program.dropout_rand.NextDouble() < this.p) {
                    output.values[i] = 0;
                    gradient_local.values[i] = 0;
                } else {
                    output.values[i] = input.values[i] * this.scaling_factor;
                    gradient_local.values[i] = this.scaling_factor;
                }
            }
            return output;
        }

        public Tensor backward(Tensor gradient_output) {

            // dL/dI
            Tensor gradient_input = new Tensor(this.I_dimensions, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            for (int i=0; i < this.I_samples * this.I_rows * this.I_columns * this.I_channels; i++) {
                
                // dL/dI = dL/dO * dO/dI
                gradient_input.values[i] = gradient_output.values[i] * this.gradient_local.values[i];
            }
            return gradient_input;
        }
    }
}
