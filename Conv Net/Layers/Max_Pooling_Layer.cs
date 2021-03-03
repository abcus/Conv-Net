using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Max_Pooling_Layer {

        int input_samples, input_rows, input_columns, input_channels;
        int filter_rows, filter_columns;
        private int output_samples, output_rows, output_columns, output_channels;

        private int input_gradient_samples, input_gradient_rows, input_gradient_columns, input_gradient_channels;

        int stride;

        Tensor input;

        public Max_Pooling_Layer (int filter_rows = 2, int filter_columns = 2, int stride = 2) {
            this.filter_rows = filter_rows;
            this.filter_columns = filter_columns;
            this.stride = stride;
        }
        
        public Tensor forward(Tensor input) {
            this.input = input;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            this.output_samples = this.input_samples;
            this.output_rows = ((this.input_rows - this.filter_rows) / this.stride) + 1;
            this.output_columns = ((this.input_columns - this.filter_columns) / this.stride + 1);
            this.output_channels = this.input_channels;

            Tensor output = new Tensor(4, this.output_samples, this.output_rows, this.output_columns, this.output_channels);

            Parallel.For(0, this.input_samples, i => {
                
                for (int j = 0; j < this.output_rows; j++) {
                    for (int k = 0; k < this.output_columns; k++) {
                        for (int l = 0; l < this.output_channels; l++) {

                            Double max = Double.MinValue;
                            
                            for (int m = 0; m < this.filter_rows; m++) {
                                for (int n = 0; n < this.filter_columns; n++) {
                                    if (this.input.values[this.input.index(i, (j * stride + m), (k * stride + n), l)] > max) {
                                        max = this.input.values[this.input.index(i, (j * stride + m), (k * stride + n), l)];
                                    }
                                }
                            }
                            output.values[output.index(i, j, k,l)] = max;
                            max = Double.MinValue;
                        }
                    }
                }
            });
            return output;
        }

        public Tensor backward(Tensor gradient_output) {

            this.input_gradient_samples = this.input_samples;
            this.input_gradient_rows = this.input_rows;
            this.input_gradient_columns = this.input_columns;
            this.input_gradient_channels = this.input_channels;

            // dL/dI
            Tensor gradient_input = new Tensor(4, this.input_gradient_samples, this.input_gradient_rows, this.input_gradient_columns, this.input_gradient_channels);

            Parallel.For(0, this.input_samples, i => {
                for (int j = 0; j <= this.input_rows - this.filter_rows; j += this.stride) {
                    for (int k = 0; k <= this.input_columns - this.filter_columns; k += this.stride) {
                        for (int l = 0; l < this.input_channels; l++) {

                            Double max = Double.MinValue;
                            int maxRow = -1;
                            int maxColumn = -1;
                            for (int m = 0; m < this.filter_rows; m++) {
                                for (int n = 0; n < this.filter_columns; n++) {
                                    if (this.input.values[this.input.index(i, (j + m), (k + n), l)] > max) {
                                        max = this.input.values[this.input.index(i, (j + m), (k + n), l)];
                                        maxRow = j + m;
                                        maxColumn = k + n;
                                    }
                                }
                            }
                            gradient_input.values[gradient_input.index(i, maxRow, maxColumn, l)] = gradient_output.values[gradient_output.index(i, (j / this.stride), (k / this.stride), l)];

                            max = Double.MinValue;
                            maxRow = -1;
                            maxColumn = -1;
                        }
                    }
                }
            });
            return gradient_input;
        }
    }
}
