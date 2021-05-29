using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;

namespace Conv_Net {
    class Utils {
        
        public static Tuple<Tensor, Tensor, Tensor, Tensor> load_MNIST(int num_train, int num_test, int num_input_rows, int num_input_columns, int num_input_channels, int num_label_rows) {
            Tensor training_images = new Tensor(4, num_train, num_input_rows, num_input_columns, num_input_channels);        
            Tensor training_labels = new Tensor(2, num_train, num_label_rows, 1, 1);

            Tensor testing_images = new Tensor(4, num_test, num_input_rows, num_input_columns, num_input_channels);
            Tensor testing_labels = new Tensor(2, num_test, num_label_rows, 1, 1);


            try {
                // Load training data
                FileStream trainImagesStream = new FileStream(@"train-images.idx3-ubyte", FileMode.Open);
                BinaryReader brTrainImages = new BinaryReader(trainImagesStream);

                FileStream trainLabelsStream = new FileStream(@"train-labels.idx1-ubyte", FileMode.Open);
                BinaryReader brTrainLabels = new BinaryReader(trainLabelsStream);

                // Read header information and discard
                int a1 = brTrainImages.ReadInt32();
                int numImages = brTrainImages.ReadInt32();
                int numRows = brTrainImages.ReadInt32();
                int numCols = brTrainImages.ReadInt32();

                int a2 = brTrainLabels.ReadInt32();
                int numLabels = brTrainLabels.ReadInt32();

                // Load image, labels, and targets into array
                for (int i = 0; i < num_train; i++) {
                    for (int j = 0; j < num_input_rows; j++) {
                        for (int k = 0; k < num_input_columns; k++) {
                            for (int l = 0; l < num_input_channels; l++) {
                                Double pixel = brTrainImages.ReadByte();
                                pixel = (-1 + (pixel / 127.5));
                                training_images.values[training_images.index(i, j, k, l)] = pixel;
                            }
                        }
                    }
                    // Load label
                    int label = brTrainLabels.ReadByte();
                    training_labels.values[training_labels.index(i, label, 0, 0)] = 1.0;
                }
                trainImagesStream.Close();
                brTrainImages.Close();

                trainLabelsStream.Close();
                brTrainLabels.Close();

                // Load test data
                FileStream testImagesStream = new FileStream(@"t10k-images.idx3-ubyte", FileMode.Open);
                BinaryReader brTestImages = new BinaryReader(testImagesStream);

                FileStream testLabelsStream = new FileStream(@"t10k-labels.idx1-ubyte", FileMode.Open);
                BinaryReader brTestLabels = new BinaryReader(testLabelsStream);

                // Read header information and discard
                a1 = brTestImages.ReadInt32();
                numImages = brTestImages.ReadInt32();
                numRows = brTestImages.ReadInt32();
                numCols = brTestImages.ReadInt32();

                a2 = brTestLabels.ReadInt32();
                numLabels = brTestLabels.ReadInt32();

                // Load image, labels, and targets into array
                for (int i = 0; i < num_test; i++) {
                    for (int j = 0; j < num_input_rows; j++) {
                        for (int k = 0; k < num_input_columns; k++) {
                            for (int l = 0; l < num_input_channels; l++) {
                                Double pixel = brTestImages.ReadByte();
                                pixel = (-1 + (pixel / 127.5));
                                testing_images.values[testing_images.index(i, j, k, l)] = pixel;
                            }
                        }
                    }
                    // Load label
                    int label = brTestLabels.ReadByte();
                    testing_labels.values[testing_labels.index(i, label, 0, 0)] = 1.0;
                }
                testImagesStream.Close();
                brTestImages.Close();

                testLabelsStream.Close();
                brTestLabels.Close();

                return Tuple.Create(training_images, training_labels, testing_images, testing_labels);
            } catch {

            }
            return null;
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
        static public void print_images(Tensor image, int image_sample) {
            
            int image_rows = image.dim_2;
            int image_columns = image.dim_3;
            int image_channels = image.dim_4; // 1

            Console.WriteLine("\t\t\t\t   ╔════════════════════════════╗");
            for (int i = 0; i < image_rows; i++) {
                Console.Write("\t\t\t\t   ║");
                for (int j = 0; j < image_columns; j++) {
                    for (int k = 0; k < image_channels; k++) {
                        Double pixel = image.values[image.index(image_sample, i, j, k)];
                        if (pixel < 0.0) {
                            Console.Write(" ");
                        } else if (pixel >= 0.0 && pixel < 0.5) {
                            Console.Write("▒");
                        } else if (pixel >= 0.5 && pixel < 0.75) {
                            Console.Write("▓");
                        } else if (pixel >= 0.75) {
                            Console.Write("█");
                        }
                    }
                }
                Console.Write("║\n");
            }
            Console.WriteLine("\t\t\t\t   ╚════════════════════════════╝");
        }

        /// <summary>
        /// Returns A * B + C
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
                    C.values[i * B_col + j] += temp;
                }
            });
            return C;
        }

        public static Tensor F_2_col(Tensor F) {
            Tensor F_2d = new Tensor(2, F.dim_1, F.dim_2 * F.dim_3 * F.dim_4);
            F_2d.values = F.values;
            return F_2d;
        }
        public static Tensor I_2_col(Tensor I, int F_rows, int F_columns, int F_channels, int stride, int dilation) {

            int I_2d_rows = F_rows * F_columns * F_channels;
            int I_samples = I.dim_1;
            int I_rows = I.dim_2;
            int I_cols = I.dim_3;
            int O_rows = (I_rows - F_rows * dilation + dilation - 1) / stride + 1;
            int O_columns = (I_cols - F_columns * dilation + dilation - 1) / stride + 1;
            int I_2d_columns = O_rows * O_columns * I_samples;

            Tensor I_2d = new Tensor(2, I_2d_rows, I_2d_columns);

            for (int i = 0; i < F_rows; i++) {
                for (int j = 0; j < F_columns; j++) {
                    for (int k = 0; k < F_channels; k++) {
                        for (int l = 0; l < I_samples; l++) {
                            for (int m = 0; m < O_rows; m++) {
                                for (int n = 0; n < O_columns; n++) {
                                    I_2d.values[(i * F_columns * F_channels + j * F_channels + k) * I_2d_columns + (l * O_rows * O_columns + m * O_columns + n)] = I.values[I.index(l, m * stride + i * dilation, n * stride + j * dilation, k)];
                                }
                            }
                        }
                    }
                }
            }
            return I_2d;
        }

        public static Tensor B_2_col(Tensor B, int I_samples, int I_rows, int I_columns, int F_rows, int F_columns, int stride, int dilation) {
            int B_2d_rows = B.dim_1;
            int O_rows = (I_rows - F_rows * dilation + dilation - 1) / stride + 1;
            int O_columns = (I_columns - F_columns * dilation + dilation - 1) / stride + 1;
            int B_2dcolumns = O_rows * O_columns * I_samples;

            Tensor B_2d = new Tensor(2, B_2d_rows, B_2dcolumns);
            for (int i = 0; i < B_2d_rows; i++) {
                for (int j = 0; j < B_2dcolumns; j++) {
                    B_2d.values[i * B_2dcolumns + j] = B.values[i];
                }
            }
            return B_2d;
        }

        public static Tensor col_2_O(Tensor O_2d, int O_sample, int O_rows, int O_columns, int O_channels) {
            Tensor O = new Tensor(4, O_sample, O_rows, O_columns, O_channels);
            for (int i = 0; i < O_sample; i++) {
                for (int j = 0; j < O_rows; j++) {
                    for (int k = 0; k < O_columns; k++) {
                        for (int l = 0; l < O_channels; l++) {
                            O.values[O.index(i, j, k, l)] = O_2d.values[l * O_sample * O_columns * O_rows + i * O_columns * O_rows + j * O_columns + k];
                        }
                    }
                }
            }
            return O;
        }

        public static Tensor dO_2_col(Tensor dO) {
            int dO_sample = dO.dim_1;
            int dO_rows = dO.dim_2;
            int dO_columns = dO.dim_3;
            int dO_channels = dO.dim_4;

            int dO_2d_rows = dO_channels;
            int dO_2d_columns = dO_sample * dO_rows * dO_columns;

            // dimensions are: (dO_sample * dO_channel) x (dO_row * dO_column)
            Tensor dO_2d = new Tensor(2, dO_2d_rows, dO_2d_columns);
            for (int i = 0; i < dO_channels; i++) {
                for (int j=0; j < dO_sample; j++) {
                    for (int k=0; k < dO_rows; k++) {
                        for (int l=0; l < dO_columns; l++) {
                            dO_2d.values[i * dO_2d_columns + (j * dO_rows * dO_columns + k * dO_columns + l)] = dO.values[dO.index(j, k, l, i)];
                        }
                    }
                }
            }
            return dO_2d;
        }
        public static Tensor I_2_col_backprop(Tensor I, int dO_rows, int dO_columns, int F_rows, int F_columns, int F_channels, int stride, int dilation) {
            int I_samples = I.dim_1;

            int I_2d_rows = I_samples * dO_rows * dO_columns;
            int I_2d_columns = F_rows * F_columns * F_channels;
            Tensor I_2d = new Tensor(2, I_2d_rows, I_2d_columns);

            for (int i = 0; i < I_samples; i++) {
                for (int j = 0; j < dO_rows; j++) {
                    for (int k = 0; k < dO_columns; k++) {
                        for (int l = 0; l < F_rows; l++) {
                            for (int m = 0; m < F_columns; m++) {
                                for (int n = 0; n < F_channels; n++) {
                                    I_2d.values[(i * dO_rows * dO_columns + j * dO_columns + k) * (I_2d_columns) + (l * F_columns * F_channels + m * F_channels + n)] = I.values[I.index(i, l * stride + j * dilation, m * stride + k * dilation, n)];
                                }
                            }
                        }
                    }
                }
            }
            return I_2d;
        }
        public static Tensor col_2_dF(Tensor dF_2d, int F_num, int F_rows, int F_columns, int F_channels) {
            Tensor dF = new Tensor(4, F_num, F_rows, F_columns, F_channels);
            dF.values = dF_2d.values;

            //for (int i=0; i < F_num; i++) {
            //    for (int j=0; j < F_rows; j++) {
            //        for (int k=0; k < F_columns; k++) {
            //            for (int l=0;l < F_channels; l++) {
            //                dF.values[dF.index(i, j, k, l)] = dF_2d.values[];
            //            }
            //        }
            //    }
            //}

            return dF;
        }

        public static Tensor F_rotated_2_col(Tensor F_rotated) {
            int F_rotated_num = F_rotated.dim_1;
            int F_rotated_rows = F_rotated.dim_2;
            int F_rotated_columns = F_rotated.dim_3;
            int F_rotated_channels = F_rotated.dim_4;
            int F_rotated_2d_columns = F_rotated_rows * F_rotated_columns * F_rotated_num;


            Tensor F_rotated_2d = new Tensor(2, F_rotated_channels, F_rotated_rows * F_rotated_columns * F_rotated_num);
            for (int i=0; i < F_rotated_channels; i++) {
                for (int j = 0; j < F_rotated_num; j++) {
                    for (int k=0; k < F_rotated_rows; k++) {
                        for (int l = 0; l < F_rotated_columns; l++) {
                            F_rotated_2d.values[(i) * (F_rotated_2d_columns) + (j * F_rotated_rows * F_rotated_columns + k * F_rotated_columns + l)] = F_rotated.values[F_rotated.index(j, k, l, i)];
                        }
                    }
                }
            }
            
            return F_rotated_2d;
        }
        public static Tensor dO_padded_2_col(Tensor dO_padded, int F_rows, int F_columns, int F_num, int I_rows, int I_columns, int dilation) {
            int dO_padded_2d_columns = I_rows * I_columns;


            Tensor dO_padded_2d = new Tensor(2, F_rows * F_columns * F_num, I_rows * I_columns);

            for (int i=0; i < F_rows; i++) {
                for (int j=0; j < F_columns; j++) {
                    for (int k=0; k < I_rows; k++) {
                        for (int l=0; l < I_columns; l++) {
                            for (int m = 0; m < F_num; m++) {
                                dO_padded_2d.values[(m * F_rows * F_columns + i * F_columns + j) * dO_padded_2d_columns + (k * I_columns + l)] = dO_padded.values[dO_padded.index(0, k + i * dilation, l + j * dilation, m)];
                                //Console.WriteLine(dO_padded.values[dO_padded.index(0, k + i, l + j, m)]);
                            }
                        }
                    }
                }
            }
            return dO_padded_2d;
        }

        static public void print_labels(Tensor label, Tensor output, int label_sample) {

            int label_rows = label.dim_2;
            int max_index_label = -1;
            Double max_value_label = Double.MinValue;

            int max_index_output = -1;
            Double max_value_output = Double.MinValue;

            for (int i=0; i < label_rows; i++) {
                if (label.values[label_sample * label_rows + i] > max_value_label) {
                    max_value_label = label.values[label_sample * label_rows + i];
                    max_index_label = i;
                }
            }

            for (int i = 0; i < label_rows; i++) {
                if (output.values[label_sample * label_rows + i] > max_value_output) {
                    max_value_output = output.values[label_sample * label_rows + i];
                    max_index_output = i;
                }
            }

            Console.WriteLine("┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐");
            Console.Write("│");

            for (int i = 0; i < label_rows; i++) {
                if (i == max_index_label) {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write("    " + i);
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write("    │");

                } else {
                    Console.Write("    " + i);
                    Console.Write("    │");
                }
            }

            Console.WriteLine("\n├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤");
            Console.Write("│");

            for (int i = 0; i < label_rows; i++) {
                if (i == max_index_output) {
                    if (max_index_output == max_index_label) {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.Write(" " + "{0:0.00000}", output.values[label_sample * label_rows + i]);
                    } else {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Write(" " + "{0:0.00000}", output.values[label_sample * label_rows + i]);
                    }
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write(" │");

                } else {
                    Console.Write(" " + "{0:0.00000}", output.values[label_sample * label_rows + i]);
                    Console.Write(" │");
                }
            }

            Console.WriteLine("\n└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘");
        }
    }
}
