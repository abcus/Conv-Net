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
        private Tensor dLocal;

        public Dropout_Layer(Double p) {
            this.p = p;
            this.scaling_factor = 1 / (1 - this.p);
        }

        public Tensor forward(Tensor I) {
            this.dLocal = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4, I.dim_5);
            
            // Do not parallelize because of random number generation
            for (int i=0; i < I.values.Count(); i++) {
                if (Program.dropout_rand.NextDouble() < this.p) {
                    I.values[i] = 0;
                    dLocal.values[i] = 0;
                } else {
                    I.values[i] = I.values[i] * this.scaling_factor;
                    dLocal.values[i] = this.scaling_factor;
                }
            }
            // O is calculated in-place from I
            return I;
        }

        public Tensor backward(Tensor dO) {
            Parallel.For(0, this.dLocal.values.Count(), i => {

                // ∂L/∂I = ∂L/∂O * ∂O/∂I
                this.dLocal.values[i] *= dO.values[i];
            });
            // dI is calculated in-place from dLocal
            return this.dLocal;
        }
    }
}
