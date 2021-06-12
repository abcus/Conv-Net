using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;

namespace Conv_Net {
    class Utils {
        
       public enum OUTPUT_TYPE: int{
            OUTPUT,
            GRADIENT_WEIGHT,
            GRADIENT_INPUT
        }

        /// <summary>
        /// Shuffles the 60000 training images and labels at the beginning of each epoch
        /// </summary>
        /// <param name="training_images"></param>
        /// <param name="training_labels"></param>
        public static void shuffle_training(Tensor training_images, Tensor training_labels) {
            int image_samples = training_images.dim_1; //60000
            int image_rows = training_images.dim_2;
            int image_columns = training_images.dim_3;
            int input_channels = training_images.dim_4;

            int label_rows= training_labels.dim_2;

            for (int i = image_samples - 1; i > 0; i--) {
                int excluded_sample = Program.rand.Next(0, i);
                
                for (int j = 0; j < image_rows; j++) {
                    for (int k = 0; k < image_columns; k++) {
                        for (int l = 0; l < input_channels; l++) {
                            (training_images.values[training_images.index(i, j, k, l)], training_images.values[training_images.index(excluded_sample, j, k, l)])
                            = (training_images.values[training_images.index(excluded_sample, j, k, l)], training_images.values[training_images.index(i, j, k, l)]);
                        }
                    }
                }
                for (int j=0; j < label_rows; j++) {
                    (training_labels.values[i * label_rows + j], training_labels.values[excluded_sample * label_rows + j]) 
                    = (training_labels.values[excluded_sample * label_rows + j], training_labels.values[i * label_rows + j]);
                }
            }
        }
        /// <summary>
        /// Generates random numbers from normal distribution, copied from MathNet Numerics (Box-Muller algorithm)
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="mean"></param>
        /// <param name="st_dev"></param>
        /// <returns></returns>
        public static double next_normal(System.Random rand, Double mean, Double st_dev) {
            Double x;
            while (!polar_transform(rand.NextDouble(), rand.NextDouble(), out x, out _)) {
            }
            return mean + (st_dev * x);
        }

        public static bool polar_transform(double a, double b, out double x, out double y) {
            var v1 = (2.0 * a) - 1.0;
            var v2 = (2.0 * b) - 1.0;
            var r = (v1 * v1) + (v2 * v2);
            if (r > 1.0 || r == 0.0) {
                x = 0;
                y = 0;
                return false;
            }
            var fac = Math.Sqrt(-2.0 * Math.Log(r) / r);
            x = v1 * fac;
            y = v2 * fac;
            return true;
        }
       

        static public Tensor elementwise_product (Tensor A, Tensor B) {
            Tensor C = new Tensor(A.dimensions, A.dim_1, A.dim_2, A.dim_3, A.dim_4);
            for (int i=0; i < C.values.Length; i++) {
                C.values[i] = A.values[i] * B.values[i]; 
            }
            return C;
        }

        static public Tensor add (Tensor A, Tensor B) {
            Tensor C = new Tensor(A.dimensions, A.dim_1, A.dim_2, A.dim_3, A.dim_4);
            for (int i=0; i < C.values.Length; i++) {
                C.values[i] = A.values[i] + B.values[i];   
            }
            return C;
        }

        static public Tensor subtract(Tensor A, Tensor B) {
            Tensor C = new Tensor(A.dimensions, A.dim_1, A.dim_2, A.dim_3, A.dim_4);
            for (int i = 0; i < C.values.Length; i++) {
                C.values[i] = A.values[i] - B.values[i];
            }
            return C;
        }

        static public Double sum (Tensor A) {
            Double sum = 0.0;
            for (int i=0; i < A.values.Length; i++) {
                sum += A.values[i];
            }
            return sum;
        }

        static public Tensor copy (Tensor A) {
            Tensor copy = new Tensor(A.dimensions, A.dim_1, A.dim_2, A.dim_3, A.dim_4);
            for (int i=0; i < A.values.Length; i++) {
                copy.values[i] = A.values[i];
            }
            return copy;
        }

        static public Double Average_L2_Distance (Tensor A, Tensor B) {
            Double distance = 0.0;

            for (int i=0; i < A.values.Length; i++) {
                distance += Math.Abs(A.values[i] - B.values[i]);
            }
            return distance/A.values.Length;
        }

        static public Tensor scalar_product(Double N, Tensor A) {
            Tensor C = new Tensor(A.dimensions, A.dim_1, A.dim_2, A.dim_3, A.dim_4);
            for (int i=0; i < C.values.Length; i++) {
                C.values[i] = A.values[i] * N;
            }
            return C;
        }

        
        /// <summary>
        /// Returns C = A * B + C
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <param name="C"></param>
        /// <returns></returns>
        static public Tensor dgemm_cs(Tensor A, Tensor B, Tensor C) {
            int A_row = A.dim_1;
            int A_col = A.dim_2;
            int B_row = B.dim_1;
            int B_col = B.dim_2;
            int B_transposed_row = B_col;
            int B_transposed_col = B_row;
            Tensor B_transposed = new Tensor(2, B_transposed_row, B_transposed_col);
            Tensor O = new Tensor(2, A_row, B_col);

            // Transpose B
            Parallel.For(0, B_transposed_row, i => {
                for (int j = 0; j < B_transposed_col; j++) {
                    B_transposed.values[i * B_transposed_col + j] = B.values[j * B_col + i];
                }
            });

            Parallel.For(0, A_row, i => {
                for (int j = 0; j < B_transposed_row; j++) {
                    Double temp = 0.0;
                    for (int k = 0; k < A_col; k++) {
                        temp += A.values[i * A_col + k] * B_transposed.values[j * B_transposed_col + k];
                    }
                    O.values[i * B_col + j] = (temp + C.values[i * B_col + j]);
                }
            });
            return O;
        }


        /// <summary>
        /// Splits one tensor into a list of tensors
        /// </summary>
        /// <param name="merge"> tensor to be split </param>
        /// <param name="dim"> dimension to aplit across </param>
        /// <param name="split_N"> number of tensors after split </param>
        /// <returns></returns>
        public static Tensor[] split(Tensor merge, int dim, int split_N) {
            Tensor[] split_list = new Tensor[split_N];
            
            if (split_N == 1) {
                split_list[0] = merge;
                return split_list;
            } else {
                int split_dim_1 = 0, split_dim_2 = 0, split_dim_3 = 0, split_dim_4 = 0;
                Tensor split;

                for (int i = 0; i < split_N; i++) {
                    switch (dim) {
                        case 1:
                            split_dim_1 = merge.dim_1 / split_N;
                            split_dim_2 = merge.dim_2;
                            split_dim_3 = merge.dim_3;
                            split_dim_4 = merge.dim_4;
                            break;
                        case 2:
                            break;
                        case 3:
                            break;
                        case 4:
                            split_dim_1 = merge.dim_1;
                            split_dim_2 = merge.dim_2;
                            split_dim_3 = merge.dim_3;
                            split_dim_4 = merge.dim_4 / split_N;
                            break;
                    }
                    split = new Tensor(4, split_dim_1, split_dim_2, split_dim_3, split_dim_4);

                    for (int j = 0; j < split_dim_1; j++) {
                        for (int k = 0; k < split_dim_2; k++) {
                            for (int l = 0; l < split_dim_3; l++) {
                                for (int m = 0; m < split_dim_4; m++) {
                                    switch (dim) {
                                        case 1:
                                            split.values[split.index(j, k, l, m)] = merge.values[merge.index(j + i * (merge.dim_1 / split_N), k, l, m)];
                                            break;
                                        case 2:
                                            break;
                                        case 3:
                                            break;
                                        case 4:
                                            split.values[split.index(j, k, l, m)] = merge.values[merge.index(j, k, l, m + i * (merge.dim_4 / split_N))];
                                            break;
                                    }
                                }
                            }
                        }
                    }
                    split_list[i] = split;
                }
                return split_list;
            }
        }

        /// <summary>
        /// Merges a list of tensors into one tensor
        /// </summary>
        /// <param name="split_list"> list of tensors to be merged </param>
        /// <param name="dim"> dimension to merge across </param>
        /// <returns></returns>
        public static Tensor merge(Tensor[] split_list, int dim) {
            int split_N = split_list.Length; int split_dim_1 = split_list[0].dim_1; int split_dim_2 = split_list[0].dim_2; int split_dim_3 = split_list[0].dim_3; int split_dim_4 = split_list[0].dim_4;
            Tensor merge;
            if (split_N == 1) {
                merge = split_list[0];
                return merge;
            } else {
                int merge_dim_1 = 0, merge_dim_2 = 0, merge_dim_3 = 0, merge_dim_4 = 0;

                switch (dim) {
                    case 1:
                        merge_dim_1 = split_dim_1 * split_N;
                        merge_dim_2 = split_dim_2;
                        merge_dim_3 = split_dim_3;
                        merge_dim_4 = split_dim_4;
                        break;
                    case 2:
                        break;
                    case 3:
                        break;
                    case 4:
                        merge_dim_1 = split_dim_1;
                        merge_dim_2 = split_dim_2;
                        merge_dim_3 = split_dim_3;
                        merge_dim_4 = split_dim_4 * split_N;
                        break;
                }
                merge = new Tensor(4, merge_dim_1, merge_dim_2, merge_dim_3, merge_dim_4);

                for (int i = 0; i < merge_dim_1; i++) {
                    for (int j = 0; j < merge_dim_2; j++) {
                        for (int k = 0; k < merge_dim_3; k++) {
                            for (int l = 0; l < merge_dim_4; l++) {
                                Tensor split;
                                switch (dim) {
                                    case 1:
                                        split = split_list[i / split_dim_1];
                                        merge.values[merge.index(i, j, k, l)] = split.values[split.index(i % split_dim_1, j, k, l)];
                                        break;
                                    case 2:
                                        break;
                                    case 3:
                                        break;
                                    case 4:
                                        split = split_list[l / split_dim_4];
                                        merge.values[merge.index(i, j, k, l)] = split.values[split.index(i, j, k, l % split_dim_4)];
                                        break;
                                }
                            }
                        }
                    }
                }
                return merge;
            }
        }


        
        /// <summary>
        /// Convolution forward propagation, converts padded input tensor into 2D matrix
        /// </summary>
        /// <param name="image"> padded input tensor</param>
        public static Tensor image_to_matrix(Tensor image, int kernel_dim_2, int kernel_dim_3, int stride, int dilation, Utils.OUTPUT_TYPE output_type) {
            Tensor image_matrix = null;
            int output_dim_2 = (image.dim_2 - kernel_dim_2 * dilation + dilation - 1) / stride + 1;
            int output_dim_3 = (image.dim_3 - kernel_dim_3 * dilation + dilation - 1) / stride + 1;

            if (output_type == Utils.OUTPUT_TYPE.OUTPUT) {
                // image_dim_1 = output_dim_1
                // image_dim_4 = kernel_dim_4
                // kernel_dim_1 = output_dim_4
                int kernel_dim_4 = image.dim_4;
                int output_dim_1 = image.dim_1;
                
                int image_matrix_rows = kernel_dim_2 * kernel_dim_3 * kernel_dim_4;
                int image_matrix_columns = output_dim_1 * output_dim_2 * output_dim_3;

                image_matrix = new Tensor(2, image_matrix_rows, image_matrix_columns);

                Parallel.For(0, kernel_dim_2, i => {
                    for (int j = 0; j < kernel_dim_3; j++) {
                        for (int m = 0; m < output_dim_2; m++) {
                            for (int n = 0; n < output_dim_3; n++) {
                                for (int k = 0; k < kernel_dim_4; k++) {
                                    for (int l = 0; l < output_dim_1; l++) {
                                        image_matrix.values[(i * kernel_dim_3 * kernel_dim_4 + j * kernel_dim_4 + k) * image_matrix_columns + (l * output_dim_2 * output_dim_3 + m * output_dim_3 + n)] = image.values[image.index(l, m * stride + i * dilation, n * stride + j * dilation, k)];
                                    }
                                }
                            }
                        }
                    }
                });
            } else if (output_type == Utils.OUTPUT_TYPE.GRADIENT_WEIGHT) {
                // image_dim_1 = kernel_dim_1
                // image_dim_4 = output_dim_4
                // kernel_dim_4 = output_dim_1
                int kernel_dim_1 = image.dim_1;
                int output_dim_4 = image.dim_4;

                int image_matrix_rows = kernel_dim_1 * kernel_dim_2 * kernel_dim_3;
                int image_matrix_columns = output_dim_2 * output_dim_3 * output_dim_4;

                image_matrix = new Tensor(2, image_matrix_rows, image_matrix_columns);

                for (int i=0; i < kernel_dim_1; i ++) {
                    for (int j = 0; j < kernel_dim_2; j++) {
                        for (int k = 0; k < kernel_dim_3; k++) {
                            for (int l = 0; l < output_dim_2; l++) {
                                for (int m = 0; m < output_dim_3; m++) {
                                    for (int n = 0; n < output_dim_4; n++) {
                                        image_matrix.values[(i * kernel_dim_2 * kernel_dim_3 + j * kernel_dim_3 + k) * (image_matrix_columns) + (l * output_dim_3 * output_dim_4 + m * output_dim_4 + n)] = image.values[image.index(i, l * stride + j * dilation, m * stride + k * dilation, n)];
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (output_type == Utils.OUTPUT_TYPE.GRADIENT_INPUT) {
                // image_dim_1 = output_dim_1
                // image_dim_4 = kernel_dim_1
                // kernel_dim_4 = output_dim_4
                int kernel_dim_1 = image.dim_4;
                int output_dim_1 = image.dim_1;

                int image_matrix_rows = kernel_dim_1 * kernel_dim_2 * kernel_dim_3;
                int image_matrix_columns = output_dim_1 * output_dim_2 * output_dim_3;

                image_matrix = new Tensor(2, image_matrix_rows, image_matrix_columns);

                for(int i = 0; i < kernel_dim_1; i++){
                    for (int j = 0; j < kernel_dim_2; j++) {
                        for (int k = 0; k < kernel_dim_3; k++) {
                            for (int l = 0; l < output_dim_1; l++) {
                                for (int m = 0; m < output_dim_2; m++) {
                                    for (int n = 0; n < output_dim_3; n++) {
                                        image_matrix.values[(i * kernel_dim_2 * kernel_dim_3 + j * kernel_dim_3 + k) * image_matrix_columns + (l * output_dim_2 * output_dim_3 + m * output_dim_3 + n)] = image.values[image.index(l, m + j * dilation, n + k * dilation, i)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return image_matrix;
        }

       

        /// <summary>
        /// Convolution forward propagation, converts bias tensor into 2D matrix with same dimensions as output (to perform elementwise sum with output)
        /// </summary>
        public static Tensor B_to_matrix(Tensor B, int I_samples, int I_rows, int I_columns, int F_rows, int F_columns, int stride, int dilation) {
            int O_rows = (I_rows - F_rows * dilation + dilation - 1) / stride + 1;
            int O_columns = (I_columns - F_columns * dilation + dilation - 1) / stride + 1;

            int B_matrix_rows = B.dim_1;
            int B_matrix_columns = I_samples * O_rows * O_columns;

            Tensor B_matrix = new Tensor(2, B_matrix_rows, B_matrix_columns);
            for (int i = 0; i < B_matrix_rows; i++) {
                for (int j = 0; j < B_matrix_columns; j++) {
                    B_matrix.values[i * B_matrix_columns + j] = B.values[i];
                }
            }
            return B_matrix;
        }

        /// <summary>
        /// Convolution forward and backward propagation, converts 2D output matrix or 2D gradient input matrix into tensor
        /// </summary>
        public static Tensor matrix_to_tensor(Tensor X_matrix, int X_sample, int X_rows, int X_columns, int X_channels) {
            Tensor X = new Tensor(4, X_sample, X_rows, X_columns, X_channels);

            Parallel.For(0, X_sample, i => {
                for (int j = 0; j < X_rows; j++) {
                    for (int k = 0; k < X_columns; k++) {
                        for (int l = 0; l < X_channels; l++) {
                            X.values[X.index(i, j, k, l)] = X_matrix.values[l * X_sample * X_rows * X_columns + i * X_rows * X_columns + j * X_columns + k];
                        }
                    }
                }
            });
            return X;
        }


        public static Tensor column_vector_1 (int rows) {
            Tensor column_vector_1 = new Tensor(2, rows, 1);
            for (int i=0; i < column_vector_1.values.Length; i++) {
                column_vector_1.values[i] = 1.0;
            }
            return column_vector_1;
        }

        public static Tensor row_vector_1 (int columns) {
            Tensor row_vector_1 = new Tensor(2, 1, columns);
            for (int i=0; i < row_vector_1.values.Length; i++) {
                row_vector_1.values[i] = 1.0;
            }
            return row_vector_1;
        }

       
        

        /// <summary>
        /// Converts a kernel tensor into a kernel matrix
        /// </summary>
        /// <param name="kernel"> kernel tensor </param>
        /// <param name="dim"> dimension of the kernel tensor that will be the row of the kernel matrix </param>
        /// <returns></returns>
        public static Tensor kernel_to_matrix(Tensor kernel, int dim) {
            int kernel_dim_1 = kernel.dim_1;
            int kernel_dim_2 = kernel.dim_2;
            int kernel_dim_3 = kernel.dim_3;
            int kernel_dim_4 = kernel.dim_4;

            int kernel_matrix_rows;
            int kernel_matrix_columns;
            Tensor kernel_matrix = null;

            if (dim == 1) {
                kernel_matrix_rows = kernel_dim_1;
                kernel_matrix_columns = kernel_dim_2 * kernel_dim_3 * kernel_dim_4;

                kernel_matrix = new Tensor(2, kernel_matrix_rows, kernel_matrix_columns);
                kernel_matrix.values = kernel.values;
            } else if (dim == 4) {
                kernel_matrix_rows = kernel_dim_4;
                kernel_matrix_columns = kernel_dim_1 * kernel_dim_2 * kernel_dim_3;
                
                kernel_matrix = new Tensor(2, kernel_matrix_rows, kernel_matrix_columns);

                Parallel.For(0, kernel_dim_4, i => {
                    for (int j = 0; j < kernel_dim_1; j++) {
                        for (int k = 0; k < kernel_dim_2; k++) {
                            for (int l = 0; l < kernel_dim_3; l++) {
                                kernel_matrix.values[i * kernel_matrix_columns + (j * kernel_dim_2 * kernel_dim_3 + k * kernel_dim_3 + l)] = kernel.values[kernel.index(j, k, l, i)];
                            }
                        }
                    }
                });
            }            
            return kernel_matrix;
        }
    
       

        /// <summary>
        /// Convolution backward propagation to calculate ∂L/∂F, converts 2D ∂L/∂F matrix into tensor
        /// </summary>
        public static Tensor dF_matrix_to_tensor(Tensor dF_matrix, int F_num, int F_rows, int F_columns, int F_channels) {
            Tensor dF = new Tensor(4, F_num, F_rows, F_columns, F_channels);
            dF.values = dF_matrix.values;
            return dF;
        }

        


        


    }
}
