using System.Linq;
using System.Threading.Tasks;

namespace Conv_Net {
    class Relu_Layer {

        // dO/dI
        Tensor dLocal; 
        public Relu_Layer() {
        }

        public Tensor forward(Tensor I) {
            this.dLocal = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4, I.dim_5);

            Parallel.For(0, I.values.Count(), i => {
                if (I.values[i] > 0) {
                    this.dLocal.values[i] = 1;
                } else {
                    I.values[i] = 0;
                    this.dLocal.values[i] = 0;
                }
            });
            // O is calculated in-place 
            return I; 
        }

        public Tensor backward (Tensor dO) {
            Parallel.For(0, this.dLocal.values.Count(), i => {
                
                // dL/dI = dL/dO * dO/dI
                this.dLocal.values[i] *= dO.values[i];
            });
            // dI is calculated in-place from dLocal
            return this.dLocal;
        }
    }
}
