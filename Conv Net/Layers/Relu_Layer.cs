using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Relu_Layer {

        // dO/dI
        Tensor dLocal; 
        public Relu_Layer() {
        }

        public Tensor forward(Tensor I) {
            dLocal = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4, I.dim_5);

            Parallel.For(0, I.values.Count(), i => {
                dLocal.values[i] = I.values[i] > 0 ? 1 : 0;
                I.values[i] = Math.Max(I.values[i], 0);
            });
            // O is calculated in-place 
            return I; 
        }

        public Tensor backward (Tensor dO) {
            Parallel.For(0, this.dLocal.values.Count(), i => {
                // dL/dI = dL/dO * dO/dI
                dLocal.values[i] *= dO.values[i];
            });
            // dI is calculated in-place from dLocal
            return dLocal;
        }
    }
}
