using System;

namespace Conv_Net {
    class Mean_Squared_Loss {

        public int I_dimensions;
        public int I_samples; 
        public int I_rows; 
        public int I_columns; 
        public int I_channels;
        public int I_elements;

        public Tensor I;
        public Tensor T;
        public Mean_Squared_Loss () {
        }

        public Tensor forward (Tensor I, Tensor T) {
            this.I = I;
            this.T = T;
            
            this.I_dimensions = I.dimensions;
            this.I_samples = I.dim_1;
            this.I_rows = I.dim_2;
            this.I_columns = I.dim_3;
            this.I_channels = I.dim_4;

            this.I_elements = this.I_rows * this.I_columns * this.I_channels;

            Tensor L = new Tensor(1, this.I_samples);

            for (int i = 0; i < this.I_samples; i ++) {
                Double loss = 0.0;
                for (int j = 0; j < this.I_elements; j++) {
                    loss += Math.Pow(I.values[i * I_elements + j] - T.values[i * I_elements + j], 2);
                }
                L.values[i] = loss / (2 * this.I_elements);
            }
            return L;
        }

        public Tensor backward () {
            int batch_size = this.I_samples;
            
            Tensor dI = new Tensor(this.I_dimensions, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            for (int i=0; i < I_samples; i++) {
                for (int j=0; j < this.I_elements; j++) {
                    dI.values[i * I_elements + j] = (I.values[i * I_elements + j] - T.values[i * I_elements + j]) / (this.I_elements * batch_size);
                }
            }
            return dI;
        }
    }
}
