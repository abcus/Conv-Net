using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class FlattenLayer {

        public FlattenLayer() {

        }

        public Double[,,] forward(Double[,,] input) {
            int input_size_x = input.GetLength(0);
            int input_size_y = input.GetLength(1);
            int input_size_z = input.GetLength(2);
            int output_size_z = input_size_x * input_size_y * input_size_z;
            Double[,,] output = new Double[1, 1, output_size_z];

            for (int input_pos_x = 0; input_pos_x < input_size_x; input_pos_x++) {
                for (int input_pos_y = 0; input_pos_y < input_size_y; input_pos_y++) {
                    for (int input_pos_z = 0; input_pos_z < input_size_z; input_pos_z++) {
                        output[0, 0, input_pos_x * input_size_y * input_size_z + input_pos_y * input_size_z + input_pos_z] = input[input_pos_x, input_pos_y, input_pos_z];
                    }
                }
            }
            return output;
        }
    }
}
