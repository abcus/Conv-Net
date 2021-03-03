using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Relu_Layer {

        Tensor input_tensor;
        public Relu_Layer() {

        }

        public Tensor forward(Tensor input) {
            this.input_tensor = input;
            Tensor output = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);

            Parallel.For(0, output.dim_1 * output.dim_2 * output.dim_3 * output.dim_4, i => {
                output.values[i] = input.values[i] >= 0 ? input.values[i] : 0;
            });
            return output;
        }

        /// <summary>
        /// Backpropagation for Relu layer
        /// </summary>
        /// <param name="gradient_output"> gradient_output = dL/dO </param>
        /// <returns></returns>
        public Tensor backward (Tensor gradient_output) {

            // dL/dI
            Tensor gradient_input = new Tensor(this.input_tensor.dimensions, this.input_tensor.dim_1, this.input_tensor.dim_2, this.input_tensor.dim_3, this.input_tensor.dim_4);
            Parallel.For(0, gradient_input.dim_1 * gradient_input.dim_2 * gradient_input.dim_3 * gradient_input.dim_4, i => {

                // dL/dI = dL/dO * dO/dI
                gradient_input.values[i] = gradient_output.values[i] * (this.input_tensor.values[i] >= 0 ? 1 : 0);
            });
            return gradient_input;
        }
    }
}
