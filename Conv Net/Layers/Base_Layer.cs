using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net{
    class Base_Layer {

        public virtual Tensor B { get; set; }
        public virtual Tensor W { get; set; }
        public virtual Tensor dB { get; set; }
        public virtual Tensor dW { get; set; }
        public virtual Tensor V_dB { get; set; }
        public virtual Tensor V_dW { get; set; }

        public virtual bool trainable_parameters { get; }
        public virtual bool test_train_mode { get; }

        public Base_Layer() {}
        public virtual Tensor forward (Tensor I) {return null;}
        public virtual Tensor forward(Tensor I, bool is_training = false) {return null;}
        public virtual Tensor backward() {return null;}
        public virtual Tensor backward(Tensor dO) {return null;}
        public virtual Tensor loss (Tensor T) { return null; }
        public virtual Tensor loss (Tensor I, Tensor T) {return null;}

    }
}
