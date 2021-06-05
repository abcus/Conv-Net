using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net{
    class Layer {

        public virtual Tensor B { get; set; }
        public virtual Tensor W { get; set; }
        public virtual Tensor dB { get; set; }
        public virtual Tensor dW { get; set; }

        public Layer() {}
        public virtual Tensor forward (Tensor I) {return null;}
        public virtual Tensor forward(Tensor I, bool is_training = false) {return null;}
        public virtual Tensor backward() {return null;}
        public virtual Tensor backward(Tensor dO) {return null;}
        public virtual Tensor loss (Tensor T) { return null; }
        public virtual Tensor loss (Tensor I, Tensor T) {return null;}

    }
}
