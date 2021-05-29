using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Tensor {
        public int dimensions;
        public int dim_1 = 1; public int dim_2 = 1; public int dim_3 = 1; public int dim_4 = 1; public int dim_5 = 1; 
        public Double[] values;

        // 1D tensor constructor
        public Tensor (int dimensions, int dim_1) {
            this.dimensions = dimensions;
            this.dim_1 = dim_1;
            values = new Double[this.dim_1];
        }

        // 2D tensor constructor
        public Tensor (int dimensions, int dim_1, int dim_2) {
            this.dimensions = dimensions;
            this.dim_1 = dim_1; this.dim_2 = dim_2;
            values = new Double[this.dim_1 * this.dim_2];
        }

        // 3D tensor constructor
        public Tensor (int dimensions, int dim_1, int dim_2, int dim_3) {
            this.dimensions = dimensions;
            this.dim_1 = dim_1; this.dim_2 = dim_2; this.dim_3 = dim_3;
            values = new Double[this.dim_1 * this.dim_2 * this.dim_3];
        }

        // 4D tensor constructor
        public Tensor (int dimensions, int dim_1, int dim_2, int dim_3, int dim_4) {
            this.dimensions = dimensions;
            this.dim_1 = dim_1; this.dim_2 = dim_2; this.dim_3 = dim_3; this.dim_4 = dim_4;
            values = new Double[this.dim_1 * this.dim_2 * this.dim_3 * this.dim_4];
        }

        /// <summary>
        /// Constructor for 5D tensor
        /// Only used to store the filter gradient in the convolutional layer backprop 
        /// </summary>
        public Tensor(int dimensions, int dim_1, int dim_2, int dim_3, int dim_4, int dim_5) {
            this.dimensions = dimensions;
            this.dim_1 = dim_1; this.dim_2 = dim_2; this.dim_3 = dim_3; this.dim_4 = dim_4; this.dim_5 = dim_5;
            values = new Double[this.dim_1 * this.dim_2 * this.dim_3 * this.dim_4 * this.dim_5];
        }

        /// <summary>
        /// Takes indices of 4D array [i, j, k, l], returns corresponding index of 1D array
        /// Faster to return 1D array index and have the caller access the element than return the element directly
        /// </summary>
        public int index(int i, int j, int k, int l) {
            return (i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l);
        }

        /// <summary>
        /// Takes indices of a 5D array [i, j, k, l, m], returns corresponding index of 1D array
        /// </summary>
        public int index(int i, int j, int k, int l, int m) {
            return ((i * this.dim_2 * this.dim_3 * this.dim_4 * this.dim_5) + (j * this.dim_3 * this.dim_4 * this.dim_5) + (k * this.dim_4 * this.dim_5) + (l * this.dim_5) + m);
        }

        /// <summary>
        /// Transposes a 2D tensor
        /// Used in backpropagation of fully connected layer to transpose weights[layer_size, previous_layer_size, 0, 0] to weights[previous_layer_size, layer_size, 0, 0]
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
        ///  Returns subset of original 4D tensor from values[image_sample_i,__,__,__] to values[(image_sample_i + image_samples),__,__,__] 
        /// Used to get batches of training images
        /// </summary>
        public Tensor subset(int image_sample_i, int image_samples) {
            Tensor subset = new Tensor(this.dimensions, image_samples, this.dim_2, this.dim_3, this.dim_4);

            Parallel.For(0, image_samples * this.dim_2 * this.dim_3 * this.dim_4, i => {
                subset.values[i] = this.values[(image_sample_i * this.dim_2 * this.dim_3 * this.dim_4) + i];
            });
            return subset;
        }

        /// <summary>
        /// Rotates tensor at values[num_filter,__,__,filter_channel] by 180 degrees (flip about X and Y axis)
        /// Used during backpropagation of convolutional layer to calculate dL/dI
        /// </summary>
        /// <returns></returns>
        public Tensor rotate_180() {
            Tensor output = new Tensor(this.dimensions, this.dim_1, this.dim_2, this.dim_3, this.dim_4);

            for (int filter = 0; filter < this.dim_1; filter++) {
                for (int i = 0; i < this.dim_2; i++) {
                    for (int j = 0; j < this.dim_3; j++) {
                        for (int k = 0; k < this.dim_4; k++) {
                            output.values[output.index(filter, i, j, k)] = this.values[this.index(filter, (this.dim_2- 1 - i), (this.dim_3 - 1 - j), k)];
                        }
                    }
                }
            }
            return output;
        }

        public Tensor dilate (int dilation) {
            if (dilation == 1) {
                return this;
            } else {
                Tensor output = new Tensor(this.dimensions, this.dim_1, (this.dim_2 - 1) * dilation + 1, (this.dim_3 - 1) * dilation + 1, this.dim_4);
                for (int i = 0; i < this.dim_1; i++) {
                    for (int j = 0; j < this.dim_2; j++) {
                        for (int k = 0; k < this.dim_3; k++) {
                            for (int l = 0; l < this.dim_4; l++) {
                                output.values[output.index(i, j * dilation, k * dilation, l)] = this.values[this.index(i, j, k, l)];
                            }
                        }
                    }
                }
                return output;
            }
        }

        /// <summary>
        /// Zero pads tensor at values [dim_1,__,__,dim_4] by pad_size
        /// Used during backpropagation of convolution layer to calculate dL/dI (also during forward propagation)
        /// </summary>
        /// <param name="pad_size"></param>
        /// <returns></returns>
        public Tensor pad(int pad_size) {
            if (pad_size == 0) {
                return this;
            } else {
                Tensor output = new Tensor(this.dimensions, this.dim_1, (this.dim_2 + (2 * pad_size)), (this.dim_3 + (2 * pad_size)), this.dim_4);
                Parallel.For(0, this.dim_1, i => {
                    for (int j = 0; j < this.dim_2; j++) {
                        for (int k = 0; k < this.dim_3; k++) {
                            for (int l = 0; l < this.dim_4; l++) {
                                output.values[output.index(i, (j + pad_size), (k + pad_size), l)] = this.values[this.index(i, j, k, l)];
                            }
                        }
                    }
                });
                return output;
            }
        }

        public Tensor unpad(int pad_size) {
            if (pad_size == 0) {
                return this;
            } else {
                Tensor output = new Tensor(this.dimensions, this.dim_1, (this.dim_2 - (2 * pad_size)), (this.dim_3 - (2 * pad_size)), this.dim_4);
                Parallel.For(0, output.dim_1, i => {
                    for (int j = 0; j < output.dim_2; j++) {
                        for (int k = 0; k < output.dim_3; k++) {
                            for (int l = 0; l < output.dim_4; l++) {
                                output.values[output.index(i, j, k, l)] = this.values[this.index(i, (j + pad_size), (k + pad_size), l)];
                            }
                        }
                    }
                });
                return output;
            }
        }

        public Tensor difference (Tensor X) {
            Tensor D = new Tensor(this.dimensions, this.dim_1, this.dim_2, this.dim_3, this.dim_4, this.dim_5);
            for (int i=0; i < this.dim_1 * this.dim_2 * this.dim_3 * this.dim_4 * this.dim_5; i++) {
                D.values[i] = this.values[i] - X.values[i];
            }
            return D;
        }

        public Tensor im_2_col(int F_rows, int F_columns, int F_channels, int dilation, int stride, int I_samples) {

            int X_rows = F_rows * F_columns * F_channels;

            int I_rows = this.dim_2;
            int I_cols = this.dim_3;
            int O_rows = (I_rows - F_rows * dilation + dilation - 1)/ stride + 1;
            int O_columns = (I_cols - F_columns * dilation + dilation - 1)/ stride + 1;
            int X_columns = O_rows * O_columns * I_samples;

            Tensor X = new Tensor(2, X_rows, X_columns);

            for (int i = 0; i < F_rows; i++) {
                for (int j = 0; j < F_columns; j++) {
                    for (int k = 0; k < F_channels; k++) {
                        for (int l = 0; l < I_samples; l++) {
                            for (int m = 0; m < O_rows; m++) {
                                for (int n = 0; n < O_columns; n++) {
                                    X.values[(i * F_columns * F_channels + j * F_channels + k) * X_columns + (l * O_rows * O_columns + m * O_columns + n)] = this.values[this.index(l, m * stride + i * dilation, n * stride + j * dilation, k)];
                                }
                            }
                        }
                    }
                }
            }
            return X;
        }

        public Tensor col_2_im(int I_sample, int I_rows, int I_columns, int I_channels) {
            Tensor X = new Tensor(4, I_sample, I_rows, I_columns, I_channels);
            for (int i=0; i < I_sample; i++) {
                for (int j=0; j < I_rows; j++) {
                    for (int k=0; k < I_columns; k++) {
                        for (int l =0; l < I_channels; l++) {
                            X.values[X.index(i, j, k, l)] = this.values[l * I_sample * I_columns * I_rows + i * I_columns * I_rows + j * I_columns + k];
                        }
                    }
                }
            }
            return X;
        }

        public Tensor mm (Tensor B) {
            int A_row = this.dim_1;
            int A_col = this.dim_2;
            int B_row = B.dim_1;
            int B_col = B.dim_2;
            int B_transposed_row = B_col;
            int B_transposed_col = B_row;
            Tensor B_transposed = new Tensor(2, B_transposed_row, B_transposed_col);
            Tensor C = new Tensor(2, A_row, B_col);

            Parallel.For(0, B_transposed_row, i => { 
                for (int j=0; j < B_transposed_col; j++) {
                    B_transposed.values[i * B_transposed_col + j] = B.values[j * B_col + i];
                }
            });

            Parallel.For(0, A_row, i => {
                for (int j = 0; j < B_transposed_row; j++) {
                    Double temp = 0.0;
                    for (int k = 0; k < A_col; k++) {
                        temp += this.values[i * A_col + k] * B_transposed.values[j * B_transposed_col + k];
                    }
                    C.values[i * B_col + j] = temp;
                }
            });
            
            return C;
        }

        public Tensor F_2_col () {
            Tensor X = new Tensor(2, this.dim_1, this.dim_2 * this.dim_3 * this.dim_4);
            X.values = this.values;
            return X;
        }

        public override string ToString() {
            if (this.dimensions == 5) {
                return "";
            }
            
            StringBuilder sb = new StringBuilder();

            if (this.dimensions == 4) {sb.Append("("); }
            else if (this.dimensions == 3) {sb.Append("[");}
            else if (this.dimensions == 2) {sb.Append("{");}
            else if (this.dimensions == 1) {sb.Append("<");}

            for (int i = 0; i < this.dim_1; i++) {
                if (this.dimensions == 4) {sb.Append("[");}
                else if (this.dimensions == 3) {sb.Append("{");} 
                else if (this.dimensions == 2) {sb.Append("<");}
                
                for (int j = 0; j < this.dim_2; j++) {
                    if (this.dimensions == 4) {sb.Append("{");}
                    else if (this.dimensions == 3) {sb.Append("<");}

                    for (int k = 0; k < this.dim_3; k++) {
                        if (this.dimensions == 4) {sb.Append("<");}

                        for (int l = 0; l < this.dim_4; l++) {
                            sb.AppendFormat("{0:0.000000}", this.values[i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l]);
                            if (this.dimensions == 4 && l < this.dim_4 - 1) {sb.Append(", ");}
                            else if (this.dimensions == 3 && k < this.dim_3 - 1) {sb.Append(", ");}
                            else if (this.dimensions == 2 && j < this.dim_2 - 1) {sb.Append(", ");}
                            else if (this.dimensions == 1 && i < this.dim_1 - 1) {sb.Append(", ");}
                        }
                        if (this.dimensions == 4) {
                            sb.Append(">");
                            if (k < this.dim_3 - 1) {sb.Append(", ");}
                        } 
                    }
                    if (this.dimensions == 4) {
                        sb.Append("}");
                        if (j < this.dim_2 - 1) {sb.Append(",\n");}
                    } else if (this.dimensions == 3) {
                        sb.Append(">");
                        if (j < this.dim_2 - 1) {sb.Append(", ");}
                    }
                }
                if (this.dimensions == 4) {
                    sb.Append("]");
                    if (i < this.dim_1 - 1) {sb.Append(",\n\n");}
                } else if (this.dimensions == 3) {
                    sb.Append("}");
                    if (i < this.dim_1 - 1) {sb.Append(",\n");}
                } else if (this.dimensions == 2) {
                    sb.Append(">");
                    if (i < this.dim_1 - 1) {sb.Append(",\n");}
                }
            }

            if (this.dimensions == 4) {sb.Append(")");}
            else if (this.dimensions == 3) {sb.Append("]");}
            else if (this.dimensions == 2) {sb.Append("}");} 
            else if (this.dimensions == 1) {sb.Append(">");}
            
            sb.Append("\n\ndimensions: " + this.dimensions);
            if (this.dimensions >= 1) {sb.Append("\ndimension 1 size: " + this.dim_1);}
            if (this.dimensions >= 2) {sb.Append("\ndimension 2 size: " + this.dim_2);}
            if (this.dimensions >= 3) {sb.Append("\ndimension 3 size: " + this.dim_3);}
            if (this.dimensions >= 4) {sb.Append("\ndimension 4 size: " + this.dim_4);}
            sb.Append("\n");
            return sb.ToString();            
        }

        public bool Equals(Tensor t) {
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
