using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Input_Layer {

        private int input_samples, input_rows, input_columns, input_channels;
        private int output_samples, output_rows, output_columns, output_channels;

        public Input_Layer() {
        }

        public Tensor forward(Tensor input) {
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            this.output_samples = this.input_samples;
            this.output_rows = this.input_rows;
            this.output_columns = this.input_columns;
            this.output_channels = this.input_channels;

            Tensor output = new Tensor(4, output_samples, output_rows, output_columns, output_channels);
            output.values = input.values;
            return output;
        }
    }
}
