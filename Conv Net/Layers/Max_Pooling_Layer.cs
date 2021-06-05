using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Max_Pooling_Layer : Layer {

        private int I_dimensions, I_samples, I_rows, I_columns, I_channels;
        private int F_rows, F_columns;
        private int O_samples, O_rows, O_columns, O_channels;
        private int stride;

        private Tensor d_local; // ∂O/∂I

        public Max_Pooling_Layer (int F_rows = 2, int F_columns = 2, int stride = 2) {
            this.F_rows = F_rows;
            this.F_columns = F_columns;
            this.stride = stride;
        }
        
        public Tensor forward(Tensor I) {
            this.I_dimensions = I.dimensions;  this.I_samples = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;
            this.d_local = new Tensor(4, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            this.O_samples = this.I_samples;
            this.O_rows = ((this.I_rows - this.F_rows) / this.stride) + 1;
            this.O_columns = ((this.I_columns - this.F_columns) / this.stride + 1);
            this.O_channels = this.I_channels;

            Tensor O = new Tensor(4, this.O_samples, this.O_rows, this.O_columns, this.O_channels);

            Parallel.For(0, this.O_samples, i => {
                for (int j = 0; j < this.O_rows; j++) {
                    for (int k = 0; k < this.O_columns; k++) {
                        for (int l = 0; l < this.O_channels; l++) {

                            Double max_value = Double.MinValue;
                            int max_row = -1; 
                            int max_column = -1;

                            for (int m = 0; m < this.F_rows; m++) {
                                for (int n = 0; n < this.F_columns; n++) {
                                    if (I.values[I.index(i, (j * stride + m), (k * stride + n), l)] > max_value) {
                                        max_value = I.values[I.index(i, (j * stride + m), (k * stride + n), l)];
                                        max_row = j * stride + m;
                                        max_column = k * stride + n;
                                    }
                                }
                            }
                            O.values[O.index(i, j, k,l)] = max_value;
                            this.d_local.values[this.d_local.index(i, max_row, max_column, l)] = 1;
                        }
                    }
                }
            });
            return O;
        }

        public Tensor backward(Tensor dO) {

            Tensor dI = new Tensor(this.I_dimensions, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

            Parallel.For(0, this.I_samples, i => {
                for (int j=0; j < this.I_rows; j ++) {
                    for (int k=0; k < this.I_columns; k++) {
                        for (int l=0; l < this.I_channels; l++) {

                            // ∂L/∂I = ∂L/∂O * ∂O/∂I 
                            dI.values[this.d_local.index(i, j, k, l)] = dO.values[dO.index(i, j / this.stride, k / this.stride, l)] * this.d_local.values[this.d_local.index(i, j, k, l)];
                        }
                    }
                }
            });
            this.d_local = null;
            return dI;
        }
    }
}
