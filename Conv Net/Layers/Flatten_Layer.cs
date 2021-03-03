using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Flatten_Layer {

        private int input_samples, input_rows, input_columns, input_channels;
        private int output_samples, output_rows, output_columns, output_channels;
        private int input_gradient_samples, input_gradient_rows, input_gradient_columns, input_gradient_channels;

        public Flatten_Layer() {
        }

        public Tensor forward(Tensor input) {
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            this.output_samples = this.input_samples;
            this.output_rows = this.input_rows * this.input_columns * this.input_channels;
            this.output_columns = 1;
            this.output_channels = 1;

            Tensor output = new Tensor(2, this.output_samples, this.output_rows, this.output_columns, this.output_channels);
            output.values = input.values;
            return output;
        }

        public Tensor backward(Tensor gradient_output) {
            this.input_gradient_samples = this.input_samples;
            this.input_gradient_rows = this.input_rows;
            this.input_gradient_columns = this.input_columns;
            this.input_gradient_channels = this.input_channels;

            Tensor gradient_input = new Tensor(4, this.input_gradient_samples, this.input_gradient_rows, this.input_gradient_columns, this.input_gradient_channels);
            gradient_input.values = gradient_output.values;
            return gradient_input;
        }
    }
}
