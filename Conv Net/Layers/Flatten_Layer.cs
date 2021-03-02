using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Flatten_Layer {

        private int numInputSamples;
        private int numInputRows;
        private int numInputColumns;
        private int numInputChannels;
        private int numOutputChannels;
        

        public Flatten_Layer() {

        }

        public Tensor forward(Tensor input) {
            this.numInputSamples = input.dim_1;
            this.numInputRows = input.dim_2;
            this.numInputColumns = input.dim_3;
            this.numInputChannels = input.dim_4;
                
            Tensor output = new Tensor(2, input.dim_1, input.dim_2 * input.dim_3 * input.dim_4, 1, 1);
            output.values = input.values;
            return output;
        }

        public Tensor backward(Tensor gradient_output) {
            Tensor gradient_input = new Tensor(4, this.numInputSamples, this.numInputRows, this.numInputColumns, this.numInputChannels);
            gradient_input.values = gradient_output.values;
            return gradient_input;
        }
    }
}
