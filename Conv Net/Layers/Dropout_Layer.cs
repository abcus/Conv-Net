using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Dropout_Layer {

        // Probability that inputs are set to 0
        private Double p;
        private Double scaling_factor;

        // ∂O/∂I
        private Tensor d_local;

        public Dropout_Layer(Double p) {
            this.p = p;
            this.scaling_factor = 1 / (1 - this.p);
        }

        public Tensor forward(Tensor I, bool is_training) {
            this.d_local = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);
            
            if (is_training) {
                // Do not parallelize because of random number generation
                for (int i = 0; i < I.values.Length; i++) {
                    if (Program.dropout_rand.NextDouble() < this.p) {
                        I.values[i] = 0;
                        d_local.values[i] = 0;
                    } else {
                        // O is calculated in-place from I
                        I.values[i] = I.values[i] * this.scaling_factor;
                        d_local.values[i] = this.scaling_factor;
                    }
                }
            } else {
                // If testing, do not change inputs
            }            
            return I;
        }

        public Tensor backward(Tensor dO) {
            Parallel.For(0, this.d_local.values.Length, i => {

                // ∂L/∂I = ∂L/∂O * ∂O/∂I
                dO.values[i] *= this.d_local.values[i];
            });
            // dI is calculated in-place from dO
            this.d_local = null;
            return dO;
        }
    }
}
