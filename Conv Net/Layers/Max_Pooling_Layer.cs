using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Max_Pooling_Layer {

        int I_samples, I_rows, I_columns, I_channels;
        int F_rows, F_columns;
        private int O_samples, O_rows, O_columns, O_channels;

        private int dI_samples, dI_rows, dI_columns, dI_channels;

        int stride;

        Tensor dLocal; // dO/dI

        public Max_Pooling_Layer (int F_rows = 2, int F_columns = 2, int stride = 2) {
            this.F_rows = F_rows;
            this.F_columns = F_columns;
            this.stride = stride;
        }
        
        public Tensor forward(Tensor I) {
            this.I_samples = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;
            this.dLocal = new Tensor(4, this.I_samples, this.I_rows, this.I_columns, this.I_channels);

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
                                    Double temp = I.values[I.index(i, (j * stride + m), (k * stride + n), l)];
                                    if (temp > max_value) {
                                        max_value = temp;
                                        max_row = j * stride + m;
                                        max_column = k * stride + n;
                                    }
                                }
                            }
                            O.values[O.index(i, j, k,l)] = max_value;
                            this.dLocal.values[this.dLocal.index(i, max_row, max_column, l)] = 1;
                        }
                    }
                }
            });
            return O;
        }

        public Tensor backward(Tensor dO) {

            this.dI_samples = this.I_samples; this.dI_rows = this.I_rows; this.dI_columns = this.I_columns; this.dI_channels = this.I_channels;
            Tensor dI = new Tensor(4, this.dI_samples, this.dI_rows, this.dI_columns, this.dI_channels);

            Parallel.For(0, this.I_samples, i => {
                for (int j=0; j < dI_rows; j ++) {
                    for (int k=0; k < dI_columns; k++) {
                        for (int l=0; l < dI_channels; l++) {
                            Double temp = this.dLocal.values[this.dLocal.index(i, j, k, l)];
                            
                            // If dO/dI (dLocal) > 0, dL/dI = dL/dO * dO/dI , otherwise dL/dI = 0
                            if (temp > 0) {
                                dI.values[dI.index(i, j, k, l)] = dO.values[dO.index(i, j / this.stride, k / this.stride, l)] * temp;
                            } else {
                                dI.values[dI.index(i, j, k, l)] = 0;
                            }
                        }
                    }
                }
            });
            return dI;
        }
    }
}
