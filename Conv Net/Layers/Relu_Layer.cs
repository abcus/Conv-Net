﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Relu_Layer {

        // ∂O/∂I
        Tensor d_local; 
        public Relu_Layer() {
        }

        public Tensor forward(Tensor I) {
            this.d_local = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4, I.dim_5);

            Parallel.For(0, I.values.Count(), i => {
                if (I.values[i] > 0) {
                    this.d_local.values[i] = 1;
                } else {
                    I.values[i] = 0;
                    this.d_local.values[i] = 0;
                }
            });
            // O is calculated in-place from I
            return I; 
        }

        public Tensor backward (Tensor dO) {
            Parallel.For(0, this.d_local.values.Count(), i => {

                // ∂L/∂I = ∂L/∂O * ∂O/∂I
                dO.values[i] *= this.d_local.values[i];
            });
            this.d_local = null;
            // dI is calculated in-place from dO
            return dO;
        }
    }
}
