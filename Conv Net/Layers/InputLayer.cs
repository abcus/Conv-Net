using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class InputLayer {

        private int input_size_x;
        private int input_size_y;
        private int input_size_z;

        public InputLayer(int input_size_x, int input_size_y, int input_size_z) {
            this.input_size_x = input_size_x;
            this.input_size_y = input_size_y;
            this.input_size_z = input_size_z;
        }

        public Double[,,] forward(Double[,,] input) {
            Double[,,] output = new Double[input_size_x, input_size_y, input_size_z];
            for (int i = 0; i < input_size_x; i++) {
                for (int j = 0; j < input_size_y; j++) {
                    for (int k = 0; k < input_size_z; k++) {
                        output[i, j, k] = input[i, j, k];
                    }
                }
            }
            return output;
        }

    }
}
