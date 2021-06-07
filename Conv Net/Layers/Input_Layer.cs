using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Input_Layer : Base_Layer {

        public Input_Layer() {
        }

        public override Tensor forward(Tensor I) {
            return I;
        }
        public override Tensor backward (Tensor dO) {
            return dO;
        }
    }
}
