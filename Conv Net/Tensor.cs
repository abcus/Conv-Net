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

        /// <summary>
        /// Takes indices of 4D array [i, j, k, l], returns corresponding index of 1D array
        /// Faster to return 1D array index and have the caller access the element than return the element directly
        /// </summary>
        public int index(int i, int j, int k, int l) {
            return (i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l);
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

        /// <summary>
        /// Pads tensor at values [dim_1,__,__,dim_4] by pad_size
        /// Used during backpropagation of convolution layer to calculate dL/dI (also during forward propagation)
        /// </summary>
        /// <param name="pad_size"></param>
        /// <returns></returns>
        public Tensor zero_pad(int pad_size) {
            Tensor output = new Tensor(this.dimensions, this.dim_1, this.dim_2 + 2 * pad_size, this.dim_3 + 2 * pad_size, this.dim_4);
            for (int filter = 0; filter < this.dim_1; filter++) {
                for (int i = 0; i < this.dim_2; i++) {
                    for (int j = 0; j < this.dim_3; j++) {
                        for (int k = 0; k < this.dim_4; k++) {
                            output.values[output.index(filter, (i + pad_size), (j + pad_size), k)] = this.values[this.index(filter, i, j, k)];
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
