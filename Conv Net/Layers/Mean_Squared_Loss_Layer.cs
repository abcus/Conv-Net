using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Mean_Squared_Loss_Layer : Layer {

        private int I_dimensions, I_samples, I_rows, I_columns, I_channels, I_elements;

        private Tensor I, T;
        public Mean_Squared_Loss_Layer () {
        }

        override public Tensor loss (Tensor I, Tensor T) {
            this.I = I;
            this.T = T;
            
            this.I_dimensions = I.dimensions; this.I_samples = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;
            this.I_elements = this.I_rows * this.I_columns * this.I_channels;

            Tensor L = new Tensor(1, 1);

            Double difference = 0.0;
            for (int i = 0; i < this.I_samples * this.I_rows * this.I_columns * this.I_channels; i ++) {
                difference += Math.Pow(I.values[i] - T.values[i], 2);
            }
            L.values[0] = difference / (this.I_samples * this.I_rows * this.I_columns * this.I_channels);
            return L;
        }

        override public Tensor backward () {
            int batch_size = this.I_samples;
            
            Tensor dI = new Tensor(this.I_dimensions, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            for (int i=0; i < this.I_samples * this.I_rows * this.I_columns * this.I_channels; i++) {
                dI.values[i] = (I.values[i] - T.values[i]) * 2 / (this.I_elements * batch_size);
            }
            this.I = null;
            this.T = null;
            return dI;
        }
    }
}
