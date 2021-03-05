using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Pad {
        private int pad_size;

        public Pad(int pad_size) {
            this.pad_size = pad_size;
        }

        public Tensor forward(Tensor input) {
            Tensor output = input.pad(pad_size);
            return output;
        }

        public Tensor backward(Tensor gradient_output) {
            Tensor gradient_input = gradient_output.unpad(pad_size);
            return gradient_input;
        }

    }
}
