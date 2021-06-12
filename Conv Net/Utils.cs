using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;

namespace Conv_Net {
    class Utils {
        
       

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


        static public Tensor forward_Conv_CPU(Convolution_Layer conv) {
            Tensor[] I_group = Utils.split_I(conv.I, conv.groups);
            Tensor[] B_group = Utils.split_W(conv.B, conv.groups);
            Tensor[] W_group = Utils.split_W(conv.W, conv.groups);
            Tensor[] O_groups = new Tensor[conv.groups];


           

            for (int i=0; i < conv.groups; i++) {
                
                Tensor B_matrix = Utils.B_to_matrix(B_group[i], conv.I_samples, conv.I_rows, conv.I_columns, conv.W_rows, conv.W_columns, conv.stride, conv.dilation);
                Tensor W_matrix = Utils.F_to_matrix(W_group[i]);
                Tensor I_matrix = Utils.I_to_matrix(I_group[i], conv.W_rows, conv.W_columns, conv.W_channels, conv.stride, conv.dilation);              
                O_groups[i] = Utils.dgemm_cs(W_matrix, I_matrix, B_matrix);
                O_groups[i] = Utils.matrix_to_tensor(O_groups[i], conv.O_samples, conv.O_rows, conv.O_columns, conv.O_channels / conv.groups);
                
            }
            Tensor output = Utils.concatenate(O_groups);
            return output;
        }

        /// <summary>
        /// Splits input/gradient output tensor for grouped convolution
        /// </summary>
        /// <param name="I"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public static Tensor[] split_I(Tensor I, int groups) {
            Tensor[] split_list = new Tensor[groups];
            for (int i = 0; i < groups; i++) {
                Tensor split_I = new Tensor(4, I.dim_1, I.dim_2, I.dim_3, I.dim_4 / groups);

                for (int j = 0; j < split_I.dim_1; j++) {
                    for (int k = 0; k < split_I.dim_2; k++) {
                        for (int l = 0; l < split_I.dim_3; l++) {
                            for (int m = 0; m < split_I.dim_4; m++) {
                                split_I.values[split_I.index(j, k, l, m)] = I.values[I.index(j, k, l, m + i * (I.dim_4 / groups))];
                            }
                        }
                    }
                }

                split_list[i] = split_I;
            }
            return split_list;
        }

        public static Tensor[] split_W (Tensor W, int groups) {
            Tensor[] split_list = new Tensor[groups];
            for (int i=0; i < groups; i++) {
                Tensor split_W = new Tensor(4, W.dim_1 / groups, W.dim_2, W.dim_3, W.dim_4);
            
                for (int j=0; j < split_W.dim_1; j++) {
                    for (int k=0; k < split_W.dim_2; k++) {
                        for (int l = 0; l < split_W.dim_3; l++) {
                            for (int m = 0; m < split_W.dim_4; m++) {
                                split_W.values[split_W.index(j, k, l, m)] = W.values[W.index(j + i * (W.dim_1 / groups), k, l, m)];
                            }
                        }
                    }
                }
                split_list[i] = split_W;
            }
            return split_list;
        }


        public static Tensor concatenate(Tensor[] T) {
            int split_tensor_samples = T[0].dim_1;
            int split_tensor_rows = T[0].dim_2;
            int split_tensor_columns = T[0].dim_3;
            int split_tensor_channels = T[0].dim_4;
            int groups = T.Length;

            Tensor concat = new Tensor(4, split_tensor_samples, split_tensor_rows, split_tensor_columns, split_tensor_channels * groups);


            for (int i = 0; i < concat.dim_1; i++) {
                for (int j = 0; j < concat.dim_2; j++) {
                    for (int k = 0; k < concat.dim_3; k++) {
                        for (int l = 0; l < concat.dim_4; l++) {
                            Tensor split = T[l / split_tensor_channels];
                            concat.values[concat.index(i, j, k, l)] = split.values[split.index(i, j, k, l % split_tensor_channels)];
                        }
                    }
                }
            }
            return concat;
        }



        /// <summary>
        /// Convolution forward propagation, converts filter tensor into 2D matrix
        /// </summary>
        public static Tensor F_to_matrix(Tensor F) {
            Tensor F_matrix = new Tensor(2, F.dim_1, F.dim_2 * F.dim_3 * F.dim_4);
            F_matrix.values = F.values;
            return F_matrix;
        }
        /// <summary>
        /// Convolution forward propagation, converts padded input tensor into 2D matrix
        /// </summary>
        /// <param name="I"> padded input tensor</param>
        public static Tensor I_to_matrix(Tensor I, int F_rows, int F_columns, int F_channels, int stride, int dilation) {          
            int I_samples = I.dim_1;
            int I_rows = I.dim_2;
            int I_cols = I.dim_3;
            int O_rows = (I_rows - F_rows * dilation + dilation - 1) / stride + 1;
            int O_columns = (I_cols - F_columns * dilation + dilation - 1) / stride + 1;
            
            int I_matrix_rows = F_rows * F_columns * F_channels;
            int I_matrix_columns = I_samples * O_rows * O_columns;

            Tensor I_matrix = new Tensor(2, I_matrix_rows, I_matrix_columns);

            Parallel.For (0, F_rows, i => {
                for (int j = 0; j < F_columns; j++) {
                    for (int k = 0; k < F_channels; k++) {
                        for (int l = 0; l < I_samples; l++) {
                            for (int m = 0; m < O_rows; m++) {
                                for (int n = 0; n < O_columns; n++) {
                                    I_matrix.values[(i * F_columns * F_channels + j * F_channels + k) * I_matrix_columns + (l * O_rows * O_columns + m * O_columns + n)] = I.values[I.index(l, m * stride + i * dilation, n * stride + j * dilation, k)];
                                }
                            }
                        }
                    }
                }
            }) ;
            return I_matrix;
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
                            X.values[X.index(i, j, k, l)] = X_matrix.values[l * X_sample * X_columns * X_rows + i * X_columns * X_rows + j * X_columns + k];
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
        /// Convolution backward propagation to calculate ∂L/∂F, converts ∂L/∂O tensor into 2D matrix
        /// </summary>
        public static Tensor dO_to_matrix(Tensor dO) {
            int dO_sample = dO.dim_1;
            int dO_rows = dO.dim_2;
            int dO_columns = dO.dim_3;
            int dO_channels = dO.dim_4;

            int dO_matrix_rows = dO_channels;
            int dO_matrix_columns = dO_sample * dO_rows * dO_columns;

            Tensor dO_matrix = new Tensor(2, dO_matrix_rows, dO_matrix_columns);

            Parallel.For(0, dO_channels, i => {
                for (int j = 0; j < dO_sample; j++) {
                    for (int k = 0; k < dO_rows; k++) {
                        for (int l = 0; l < dO_columns; l++) {
                            dO_matrix.values[i * dO_matrix_columns + (j * dO_rows * dO_columns + k * dO_columns + l)] = dO.values[dO.index(j, k, l, i)];
                        }
                    }
                }
            });
            return dO_matrix;
        }

        /// <summary>
        /// Convolution backward propagation to calculate ∂L/∂F, converts I tensor into 2D matrix
        /// </summary>
        public static Tensor I_to_matrix_backprop(Tensor I, int dO_rows, int dO_columns, int F_rows, int F_columns, int F_channels, int stride, int dilation) {
            int I_samples = I.dim_1;

            int I_matrix_rows = I_samples * dO_rows * dO_columns;
            int I_matrix_columns = F_rows * F_columns * F_channels;
            Tensor I_matrix = new Tensor(2, I_matrix_rows, I_matrix_columns);

            Parallel.For(0, I_samples, i => {
                for (int j = 0; j < dO_rows; j++) {
                    for (int k = 0; k < dO_columns; k++) {
                        for (int l = 0; l < F_rows; l++) {
                            for (int m = 0; m < F_columns; m++) {
                                for (int n = 0; n < F_channels; n++) {
                                    I_matrix.values[(i * dO_rows * dO_columns + j * dO_columns + k) * (I_matrix_columns) + (l * F_columns * F_channels + m * F_channels + n)] = I.values[I.index(i, l * stride + j * dilation, m * stride + k * dilation, n)];
                                }
                            }
                        }
                    }
                }
            });
            return I_matrix;
        }

        /// <summary>
        /// Convolution backward propagation to calculate ∂L/∂F, converts 2D ∂L/∂F matrix into tensor
        /// </summary>
        public static Tensor dF_matrix_to_tensor(Tensor dF_matrix, int F_num, int F_rows, int F_columns, int F_channels) {
            Tensor dF = new Tensor(4, F_num, F_rows, F_columns, F_channels);
            dF.values = dF_matrix.values;
            return dF;
        }

        /// <summary>
        /// Convolution backward propagation to calculate ∂L/∂I, converts 180 rotated F tensor into 2D matrix
        /// </summary>
        public static Tensor F_rotated_to_matrix(Tensor F_rotated) {
            int F_rotated_num = F_rotated.dim_1;
            int F_rotated_rows = F_rotated.dim_2;
            int F_rotated_columns = F_rotated.dim_3;
            int F_rotated_channels = F_rotated.dim_4;

            int F_rotated_matrix_rows = F_rotated_channels;
            int F_rotated_matrix_columns = F_rotated_num * F_rotated_rows * F_rotated_columns;

            Tensor F_rotated_matrix = new Tensor(2, F_rotated_matrix_rows, F_rotated_matrix_columns);

            Parallel.For(0, F_rotated_channels, i => {
                for (int j = 0; j < F_rotated_num; j++) {
                    for (int k = 0; k < F_rotated_rows; k++) {
                        for (int l = 0; l < F_rotated_columns; l++) {
                            F_rotated_matrix.values[(i) * (F_rotated_matrix_columns) + (j * F_rotated_rows * F_rotated_columns + k * F_rotated_columns + l)] = F_rotated.values[F_rotated.index(j, k, l, i)];
                        }
                    }
                }
            });
            return F_rotated_matrix;
        }


        /// <summary>
        /// Convolution backward propagation to calculate ∂L/∂I, converts dilated, padded ∂L/∂O tensor into 2D matrix
        /// </summary>
        /// <param name="dI_rows"> rows of input gradient after padding is removed </param>
        /// <param name="dI_columns"> columns of input gradient after padding is removed </param>
        /// <returns></returns>
        public static Tensor dO_dilated_padded_to_matrix(Tensor dO_dilated_padded, int F_num, int F_rows, int F_columns, int dI_samples, int dI_rows, int dI_columns, int dilation) {
            int dO_dilated_padded_matrix_rows = F_num * F_rows * F_columns;
            int dO_dilated_padded_matrix_columns = dI_samples * dI_rows * dI_columns;

            Tensor dO_dilated_padded_matrix = new Tensor(2, dO_dilated_padded_matrix_rows, dO_dilated_padded_matrix_columns);

            Parallel.For(0, F_num, i => {
                for (int j = 0; j < F_rows; j++) {
                    for (int k = 0; k < F_columns; k++) {
                        for (int l = 0; l < dI_samples; l++) {
                            for (int m = 0; m < dI_rows; m++) {
                                for (int n = 0; n < dI_columns; n++) {
                                    dO_dilated_padded_matrix.values[(i * F_rows * F_columns + j * F_columns + k) * dO_dilated_padded_matrix_columns + (l * dI_rows * dI_columns + m * dI_columns + n)] = dO_dilated_padded.values[dO_dilated_padded.index(l, m + j * dilation, n + k * dilation, i)];
                                }
                            }
                        }
                    }
                }
            });
            return dO_dilated_padded_matrix;
        }


    }
}
