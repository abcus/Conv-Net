using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;

namespace Conv_Net {
    static class Program {
        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

        }
    }

    class Convolution {

        private int num_filters;
        private int filter_size;
        private int stride;
        private Double[,,,] filter;
        private Double[] biases;
        

        public Convolution(int input_z, int num_filters, int filter_size, int stride) {

            this.num_filters = num_filters;
            this.filter_size = filter_size;
            this.stride = stride;

            filter = new Double[num_filters, filter_size, filter_size, input_z];
            // Initialize filter weights

            biases = new Double[num_filters];
        }

        private Double[,,] forward (Double[,,] input) {
            int input_x = input.GetLength(0);
            int input_y = input.GetLength(1);
            int input_z = input.GetLength(2);

            int output_x = (input_x - filter_size) / stride + 1;
            int output_y = (input_y - filter_size) / stride + 1;
            int output_z = num_filters;

            Double[,,] output = new Double[output_x, output_y, output_z];

            Double dot_product = 0.0;

            for (int filter_index = 0; filter_index < num_filters; filter_index++) {
                for (int input_x_pos = 0; input_x_pos <= input_x - filter_size; input_x_pos += stride) {
                    for (int input_y_pos = 0; input_y_pos <= input_y - filter_size; input_y_pos += stride) {
                        for (int filter_x_pos = 0; filter_x_pos < filter_size; filter_x_pos++) {
                            for (int filter_y_pos = 0; filter_y_pos < filter_size; filter_y_pos++) {
                                for (int filter_z_pos = 0; filter_z_pos < input_z; filter_z_pos++) {
                                    dot_product += filter[filter_index, filter_x_pos, filter_y_pos, filter_z_pos] * input[input_x_pos, input_y_pos, filter_z_pos];
                                }
                            }
                        }
                        dot_product += biases[filter_index];
                        output[input_x_pos / stride, input_y_pos / stride, filter_index] = dot_product;
                        dot_product = 0.0;
                    }
                }
            }
            return output;
        }
    }
}

/* To Do:
Padding
ADAM
Regularization (dropout, L2)
Batch normalization
 */


// Forward convolusion











