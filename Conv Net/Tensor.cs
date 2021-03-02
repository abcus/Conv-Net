using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Tensor {

        public int dimensions;
        public int dim_1;
        public int dim_2;
        public int dim_3;
        public int dim_4;
        public Double[] values;

        public Tensor (int dimensions, int dim_1, int dim_2, int dim_3, int dim_4) {
            this.dimensions = dimensions;
            this.dim_1 = dim_1;
            this.dim_2 = dim_2;
            this.dim_3 = dim_3;
            this.dim_4 = dim_4;
            values = new Double[this.dim_1 * this.dim_2 * this.dim_3 * this.dim_4];
        }

        public Double get (int i, int j, int k, int l) {
            return values[i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l];
        }

        public void set (int i, int j, int k, int l, Double value) {
            values[i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l] = value;
        }

        /// <summary>
        /// Transposes a 2D tensor
        /// </summary>
        public Tensor transpose_2D () {
            Tensor output = new Tensor(this.dimensions, this.dim_2, this.dim_1, this.dim_3, this.dim_4);
            for (int i = 0; i < output.dim_1; i++) {
                for (int j = 0; j < output.dim_2; j++) {
                    output.values[i * output.dim_2 + j] = this.values[j * this.dim_2 + i];
                }
            }
            return output;
        }

        /// <summary>
        ///  Returns subset of original 4D tensor which starts at values[dim_1_i * this.dim_2 * this.dim_3 * this.dim_4] and has size [dim_1_size * this.dim_2 * this.dim_3 * this.dim_4]
        /// </summary>
        public Tensor subset(int dim_1_i, int dim_1_size) {
            Tensor t = new Tensor(this.dimensions, dim_1_size, this.dim_2, this.dim_3, this.dim_4);

            Parallel.For(0, dim_1_size * this.dim_2 * this.dim_3 * this.dim_4, i => {
                t.values[i] = this.values[(dim_1_i * this.dim_2 * this.dim_3 * this.dim_4) + i];
            });
            return t;
        }


        public Tensor rotate_180() {
            Tensor output = new Tensor(this.dimensions, this.dim_1, this.dim_2, this.dim_3, this.dim_4);

            for (int filter = 0; filter < this.dim_1; filter++) {
                for (int i = 0; i < this.dim_2; i++) {
                    for (int j = 0; j < this.dim_3; j++) {
                        for (int k = 0; k < this.dim_4; k++) {
                            output.values[filter * (this.dim_2 * this.dim_3 * this.dim_4) + i * (this.dim_3 * this.dim_4) + j * (this.dim_4) + k] 
                                = this.values[filter * (this.dim_2 * this.dim_3 * this.dim_4) + (this.dim_2- 1 - i) * (this.dim_3 * this.dim_4) + (this.dim_3 - 1 - j) * (this.dim_4) + k];
                        }
                    }
                }
            }
            return output;
        }

        public Tensor zero_pad(int pad_size) {
            Tensor output = new Tensor(this.dimensions, this.dim_1, this.dim_2 + 2 * pad_size, this.dim_3 + 2 * pad_size, this.dim_4);
            for (int filter = 0; filter < this.dim_1; filter++) {
                for (int i = 0; i < this.dim_2; i++) {
                    for (int j = 0; j < this.dim_3; j++) {
                        for (int k = 0; k < this.dim_4; k++) {
                            output.values[filter * (output.dim_2 * output.dim_3 * output.dim_4) + (i + pad_size) * (output.dim_3 * output.dim_4) + (j + pad_size) * (output.dim_4) + k] 
                                = this.values[filter * (this.dim_2 * this.dim_3 * this.dim_4) + i * (this.dim_3 * this.dim_4) + j * (this.dim_4) + k];
                        }
                    }
                }
            }
            return output;
        }


        public override string ToString() {
            StringBuilder sb = new StringBuilder();

            if (dimensions == 0) {
                
                sb.AppendFormat("{0:0.00000}", this.values[0]);
                sb.Append("\n");
                sb.Append("dimensions: " + this.dimensions + "\n");

            } else if (dimensions == 1) {
                sb.Append("<");
                for (int i = 0; i < this.dim_1; i++) {
                    sb.AppendFormat("{0:0.00000}", this.values[i]);
                    if (i < this.dim_1 - 1) sb.Append(", ");
                }
                sb.Append(">");
                sb.Append("\n");
                sb.Append("dimensions: " + this.dimensions + "\nsize: " + this.dim_1 + "\n");
            } 
            else if (dimensions == 2) {
                sb.Append("{");
                for (int i = 0; i < this.dim_1; i++) {
                    sb.Append("<");
                    for (int j=0; j < this.dim_2; j++) {
                        sb.AppendFormat("{0:0.00000}", this.values[i * dim_2 + j]);
                        if (j < this.dim_2 - 1) sb.Append(", ");
                    }
                    sb.Append(">");
                    if (i < this.dim_1 - 1) sb.Append(",\n");
                }
                sb.Append("}");
                sb.Append("\n");
                sb.Append("dimensions: " + this.dimensions + "\nrows: " + this.dim_1 + "\ncolumns: " + this.dim_2 + "\n");
            } 
            else if (dimensions == 3) {
                sb.Append("[");
                for (int i=0; i < this.dim_1; i++) {
                    sb.Append("{");
                    for (int j=0; j < this.dim_2; j++) {
                        sb.Append("<");
                        for (int k=0; k < this.dim_3; k++) {
                            sb.AppendFormat("{0:0.00000}", this.values[i * this.dim_2 * this.dim_3 + j * this.dim_3 + k]);
                            if (k < this.dim_3 - 1) sb.Append(", ");
                        }
                        sb.Append(">");
                        if (j < this.dim_2 - 1) sb.Append(", ");
                    }
                    sb.Append("}");
                    if (i < this.dim_1 - 1) sb.Append(",\n");
                }
                sb.Append("]");
                sb.Append("\n");
                sb.Append("dimensions: " + this.dimensions + "\nrows: " + this.dim_1 + "\ncolumns: " + this.dim_2 + "\nchannels: " + this.dim_3 + "\n");
            } 
            else if (dimensions == 4) {
                sb.Append("(");
                for (int i=0; i < this.dim_1; i++) {
                    sb.Append("[");
                    for (int j=0; j < this.dim_2; j++) {
                        sb.Append("{");
                        for (int k=0; k < this.dim_3; k++) {
                            sb.Append("<");
                            for (int l=0; l < this.dim_4; l++) {
                                sb.AppendFormat("{0:0.00000}", this.values[i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l]);
                                if (l < this.dim_4 - 1) sb.Append(", ");
                            }
                            sb.Append(">");
                            if (k < this.dim_3 - 1) sb.Append(", ");
                        }
                        sb.Append("}");
                        if (j < this.dim_2 - 1) sb.Append(",\n");
                    }
                    sb.Append("]");
                    if (i < this.dim_1 - 1) sb.Append(",\n\n");
                }
                sb.Append(")");
                sb.Append("\n");
                sb.Append("dimensions: " + this.dimensions + "\nsamples: " + this.dim_1 + "\nrows: " + this.dim_2 + "\ncolumns: " + this.dim_3 + "\nchannels: " + this.dim_4 + "\n");
            }
            
            return sb.ToString();            
        }

        public bool Equals(Tensor t) {
            bool equals = true;
            if (this.dimensions != t.dimensions 
                || this.dim_1 != t.dim_1 
                || this.dim_2 != t.dim_2 
                || this.dim_3 != t.dim_3 
                || this.dim_4 != t.dim_4) {
                return false;
            }
            for (int i = 0; i < this.dim_1 * this.dim_2 * this.dim_3 * this.dim_4; i++) {
                if (this.values[i] != t.values[i]) {
                    return false;
                }
            }
            return true;
        }
    }
}
