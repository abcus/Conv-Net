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

        private int input_dimensions, input_samples, input_rows, input_columns, input_channels;
        private int output_samples, output_rows, output_columns, output_channels;

        // dO/dI
        private Tensor gradient_local;

        public Dropout_Layer(Double p) {
            this.p = p;
            this.scaling_factor = 1 / (1 - this.p);
        }

        public Tensor forward(Tensor input) {
            
            this.input_dimensions = input.dimensions;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            this.gradient_local = new Tensor(this.input_dimensions, this.input_samples, this.input_rows, this.input_columns, this.input_channels);
            Tensor output = new Tensor(this.input_dimensions, this.input_samples, this.input_rows, this.input_columns, this.input_channels);

            for (int i=0; i < this.input_samples * this.input_rows * this.input_columns * this.input_channels; i++) {
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
            Tensor gradient_input = new Tensor(this.input_dimensions, this.input_samples, this.input_rows, this.input_columns, this.input_channels);

            for (int i=0; i < this.input_samples * this.input_rows * this.input_columns * this.input_channels; i++) {
                
                // dL/dI = dL/dO * dO/dI
                gradient_input.values[i] = gradient_output.values[i] * this.gradient_local.values[i];
            }
            return gradient_input;
        }
    }
}
