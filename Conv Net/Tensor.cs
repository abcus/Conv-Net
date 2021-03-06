﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Tensor {
        public int dimensions;
        public int dim_1 = 1; public int dim_2 = 1; public int dim_3 = 1; public int dim_4 = 1; 
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
        public bool Equals(Tensor t) {
            if (this.dimensions != t.dimensions
                || this.dim_1 != t.dim_1
                || this.dim_2 != t.dim_2
                || this.dim_3 != t.dim_3
                || this.dim_4 != t.dim_4) {
                return false;
            }
            for (int i = 0; i < this.dim_1 * this.dim_2 * this.dim_3 * this.dim_4; i++) {
                if ((float)this.values[i] != (float)t.values[i]) {
                    return false;
                }
            }
            return true;
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
                            sb.AppendFormat("{0:0.0000000000000000}", this.values[i * this.dim_2 * this.dim_3 * this.dim_4 + j * this.dim_3 * this.dim_4 + k * this.dim_4 + l]);
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
    }
}
