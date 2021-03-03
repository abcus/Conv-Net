using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Relu_Layer {

        private int input_dimensions, input_samples, input_rows, input_columns, input_channels;
        private int output_dimensions, output_samples, output_rows, output_columns, output_channels;

        private int input_gradient_dimensions, input_gradient_samples, input_gradient_rows, input_gradient_columns, input_gradient_channels;

        Tensor input;
        public Relu_Layer() {
        }

        public Tensor forward(Tensor input) {
            this.input = input;
            this.input_dimensions = input.dimensions;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            this.output_dimensions = this.input_dimensions;
            this.output_samples = this.input_samples;
            this.output_rows = this.input_rows;
            this.output_columns = this.input_columns;
            this.output_channels = this.input_channels;

            Tensor output = new Tensor(this.output_dimensions, this.output_samples, this.output_rows, this.output_columns, this.output_channels);

            Parallel.For(0, this.output_samples * this.output_rows * this.output_columns * this.output_channels, i => {
                output.values[i] = this.input.values[i] >= 0 ? this.input.values[i] : 0;
            });
            return output;
        }

        /// <summary>
        /// Backpropagation for Relu layer
        /// </summary>
        /// <param name="gradient_output"> gradient_output = dL/dO </param>
        /// <returns></returns>
        public Tensor backward (Tensor gradient_output) {

            this.input_gradient_dimensions = this.input_dimensions;
            this.input_gradient_samples = this.input_samples;
            this.input_gradient_rows = this.input_rows;
            this.input_gradient_columns = this.input_columns;
            this.input_gradient_channels = this.input_channels;

            // dL/dI
            Tensor gradient_input = new Tensor(this.input_gradient_dimensions, this.input_gradient_samples, this.input_gradient_rows, this.input_gradient_columns, this.input_gradient_channels);
            
            Parallel.For(0, this.input_gradient_samples * this.input_gradient_rows * this.input_gradient_columns * this.input_gradient_channels, i => {

                // dL/dI = dL/dO * dO/dI
                gradient_input.values[i] = gradient_output.values[i] * (this.input.values[i] >= 0 ? 1 : 0);
            });
            return gradient_input;
        }
    }
}
