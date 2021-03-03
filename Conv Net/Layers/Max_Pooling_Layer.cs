using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Max_Pooling_Layer {

        Tensor input_tensor;
        int input_samples;
        int input_rows;
        int input_columns;
        int input_channels;

        int filter_rows;
        int filter_columns;
        int stride;

        public Max_Pooling_Layer (int filter_rows = 2, int filter_columns = 2, int stride = 2) {
            this.filter_rows = filter_rows;
            this.filter_columns = filter_columns;
            this.stride = stride;
        }
        
        public Tensor forward_tensor(Tensor input) {
            this.input_tensor = input;
            this.input_samples = input.dim_1;
            this.input_rows = input.dim_2;
            this.input_columns = input.dim_3;
            this.input_channels = input.dim_4;

            int output_rows = ((this.input_rows - this.filter_rows) / this.stride) + 1;
            int output_columns = ((this.input_columns - this.filter_columns) / this.stride + 1);
            int output_channels = this.input_channels;

            Tensor output = new Tensor(4, this.input_samples, output_rows, output_columns, output_channels);
            Double max = Double.MinValue;
            
            for (int i = 0; i < this.input_samples; i++) {
                for (int j = 0; j < output_rows; j++) {
                    for (int k = 0; k < output_columns; k++) {
                        for (int l = 0; l < output_channels; l++) {
                            for (int m = 0; m < this.filter_rows; m++) {
                                for (int n = 0; n < this.filter_columns; n++) {
                                    if (input_tensor.values[i * (this.input_tensor.dim_2 * this.input_tensor.dim_3 * this.input_tensor.dim_4) + (j * stride + m) * (this.input_tensor.dim_3 * this.input_tensor.dim_4) + (k * stride + n) * (this.input_tensor.dim_4) + l] > max) {
                                        max = input_tensor.values[i * (this.input_tensor.dim_2 * this.input_tensor.dim_3 * this.input_tensor.dim_4) + (j * stride + m) * (this.input_tensor.dim_3 * this.input_tensor.dim_4) + (k * stride + n) * (this.input_tensor.dim_4) + l];
                                    }
                                }
                            }
                            output.values[i * (output.dim_2 * output.dim_3 * output.dim_4) + j * (output.dim_3 * output.dim_4) + k * (output.dim_4) + l] = max;
                            max = Double.MinValue;
                        }
                    }
                }
            }
            return output;
        }

        public Tensor backward_tensor(Tensor gradient_output) {
            Double max = Double.MinValue;
            int maxRow = -1;
            int maxColumn = -1;

            // dL/dI
            Tensor gradient_input = new Tensor(4, this.input_samples, this.input_rows, this.input_columns, this.input_channels);

            for (int i = 0; i < this.input_samples; i ++ ) {
                for (int j = 0; j <= this.input_rows - this.filter_rows; j += this.stride) {
                    for (int k = 0; k <= this.input_columns - this.filter_columns; k += this.stride) {
                        for (int l = 0; l < this.input_channels; l++) {
                            for (int m = 0; m < this.filter_rows; m++) {
                                for (int n = 0; n < this.filter_columns; n++) {
                                    if (this.input_tensor.values[i * (this.input_tensor.dim_2 * this.input_tensor.dim_3 * this.input_tensor.dim_4) + (j + m) * (this.input_tensor.dim_3 * this.input_tensor.dim_4) + (k + n) * (this.input_tensor.dim_4) + l] > max) {
                                        max = input_tensor.values[i * (this.input_tensor.dim_2 * this.input_tensor.dim_3 * this.input_tensor.dim_4) + (j + m) * (this.input_tensor.dim_3 * this.input_tensor.dim_4) + (k + n) * (this.input_tensor.dim_4) + l];
                                        maxRow = j + m;
                                        maxColumn = k + n;
                                    }
                                }
                            }
                            gradient_input.values[i * (gradient_input.dim_2 * gradient_input.dim_3 * gradient_input.dim_4) + maxRow * (gradient_input.dim_3 * gradient_input.dim_4) + maxColumn * (gradient_input.dim_4) + l] = gradient_output.values[i * (gradient_output.dim_2 * gradient_output.dim_3 * gradient_output.dim_4) + (j / this.stride) * (gradient_output.dim_3 * gradient_output.dim_4) + (k / this.stride) * (gradient_output.dim_4) + l];

                            max = Double.MinValue;
                            maxRow = -1;
                            maxColumn = -1;
                        }
                    }
                }
            }
            return gradient_input;
        }
    }
}
