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
                        if (pixel == -1) {
                            Console.Write(".");
                        } else if (pixel == 1) {
                            Console.Write("%");
                        } else { 
                            Console.Write("o"); 
                        }
                    }
                }
                Console.Write("║\n");
            }
            Console.WriteLine("\t\t\t\t   ╚════════════════════════════╝");
        }

        static public void print_labels(Tensor label, int label_sample) {

            int label_rows = label.dim_2;
            int max_index = -1;
            Double max_value = Double.MinValue;

            for (int i=0; i < label_rows; i++) {
                if (label.values[label_sample * label_rows + i] > max_value) {
                    max_value = label.values[label_sample * label_rows + i];
                    max_index = i;
                }
            }
            
            
            Console.WriteLine("┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐");
            Console.Write("│");

            for (int i = 0; i < label_rows; i++) {
                if (i == max_index) {
                    Console.ForegroundColor = ConsoleColor.Red;
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

            for (int i=0; i < label_rows; i++) {
                if (i == max_index) {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Write(" " + "{0:0.00000}", label.values[label_sample * label_rows + i]);
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write(" │");
                    
                } else {
                    Console.Write(" " + "{0:0.00000}", label.values[label_sample * label_rows + i]);
                    Console.Write(" │");
                }
            }

            Console.WriteLine("\n└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘");
        }
    }
}
