﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;

namespace Conv_Net {
    class Utils {

        static public Double[,,] elementwiseProduct(Double[,,] x, Double[,,] y) {
            Debug.Assert(x.GetLength(2) == y.GetLength(2));
            int size = x.GetLength(2);
            Double[,,] output = new Double[1, 1, size];
            for (int i = 0; i < size; i++) {
                output[0, 0, i] = x[0, 0, i] * y[0, 0, i];
            }
            return output;
        }


        static public Double dotProduct(Double[,,] x, Double[,,] y) {
            Debug.Assert(x.GetLength(2) == y.GetLength(2));
            Double dotProduct = 0.0;
            for (int i = 0; i < x.GetLength(2); i++) {
                dotProduct += (x[0, 0, i] * y[0, 0, i]);
            }
            return dotProduct;
        }


        static public int indexMaxValue(Double[,,] input) {
            Double max = Double.MinValue;
            int index = -1;
            for (int i = 0; i < input.GetLength(2); i++) {
                if (input[0, 0, i] > max) {
                    max = input[0, 0, i];
                    index = i;
                }
            }
            return index;
        }

        static public int indexMaxValue_tensor(Tensor input) {
            Double max = Double.MinValue;
            int index = -1;
            for (int i=0; i < input.dim_2; i++) {
                if (input.values[i] > max) {
                    max = input.values[i];
                    index = i;
                }
            }
            return index;
        }

        static public Double maxValue(Double[,,] input) {
            Double max = Double.MinValue;
            for (int i = 0; i < input.GetLength(2); i++) {
                if (input[0, 0, i] > max) {
                    max = input[0, 0, i];
                }
            }
            return max;
        }

        static public Double[][,,] transpose(Double[][,,] input) {
            int x = input.Count();
            int y = input[0].GetLength(2);
            Double[][,,] output = new Double[y][,,];

            for (int i = 0; i < y; i++) {
                output[i] = new Double[1, 1, x];
                for (int j = 0; j < x; j++) {
                    output[i][0, 0, j] = input[j][0, 0, i];
                }
            }
            return output;
        }

        static public Double[,,] zeroPad(int padSize, Double[,,] input) {
            int numInputRows = input.GetLength(0);
            int numInputColumns = input.GetLength(1);
            int numInputChannels = input.GetLength(2);

            Double[,,] output = new Double[numInputRows + 2 * padSize, numInputColumns + 2 * padSize, numInputChannels];

            for (int i = 0; i < numInputRows; i++) {
                for (int j = 0; j < numInputColumns; j++) {
                    for (int k = 0; k < numInputChannels; k++) {
                        output[i + padSize, j + padSize, k] = input[i, j, k];
                    }
                }
            }
            return output;
        }

        static public Double[,,] rotate180(Double[,,] input) {
            int numInputRows = input.GetLength(0);
            int numInputColumns = input.GetLength(1);
            int numInputChannels = input.GetLength(2);

            Double[,,] output = new Double[numInputRows, numInputColumns, numInputChannels];

            for (int i = 0; i < numInputRows; i++) {
                for (int j = 0; j < numInputColumns; j++) {
                    for (int k = 0; k < numInputChannels; k++) {
                        output[i, j, k] = input[numInputRows - 1 - i, numInputColumns - 1 - j, k];
                    }
                }
            }
            return output;
        }


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
                                training_images.set(i, j, k, l, pixel);
                            }
                        }
                    }
                    // Load label
                    int label = brTrainLabels.ReadByte();
                    training_labels.set(i, label, 0, 0, 1.0);
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
                                testing_images.set(i, j, k, l, pixel);
                            }
                        }
                    }
                    // Load label
                    int label = brTestLabels.ReadByte();
                    testing_labels.set(i, label, 0, 0, 1.0);
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

        public static void shuffleTrainingSet() {
            for (int i = 60000 - 1; i > 0; i--) {
                int excluded_sample = Program.rand.Next(0, i);
                
                (Program.trainImageArray[i], Program.trainImageArray[excluded_sample]) = (Program.trainImageArray[excluded_sample], Program.trainImageArray[i]);
                (Program.trainLabelArray[i], Program.trainLabelArray[excluded_sample]) = (Program.trainLabelArray[excluded_sample], Program.trainLabelArray[i]);
            }
        }


        public static void shuffle_Tensor(Tensor training_images, Tensor training_labels) {
            int num_samples = training_images.dim_1;
            int num_image_rows = training_images.dim_2;
            int num_image_columns = training_images.dim_3;
            int num_image_channels = training_images.dim_4;
            Double[] image_data = training_images.values;

            int num_label_rows= training_labels.dim_2;
            Double[] label_data = training_labels.values;

            for (int i = num_samples - 1; i > 0; i--) {
                int excluded_sample = Program.rand.Next(0, i);
                
                for (int j = 0; j < num_image_rows; j++) {
                    for (int k = 0; k < num_image_columns; k++) {
                        for (int l = 0; l < num_image_channels; l++) {
                            (training_images.values[i * num_image_rows * num_image_columns * num_image_channels + j * num_image_columns * num_image_channels + k * num_image_channels + l], training_images.values[excluded_sample * num_image_rows * num_image_columns * num_image_channels + j * num_image_columns * num_image_channels + k * num_image_channels + l]) =
                            (training_images.values[excluded_sample * num_image_rows * num_image_columns * num_image_channels + j * num_image_columns * num_image_channels + k * num_image_channels + l], training_images.values[i * num_image_rows * num_image_columns * num_image_channels + j * num_image_columns * num_image_channels + k * num_image_channels + l]);
                        }
                    }
                }
                for (int j=0; j < num_label_rows; j++) {
                    (training_labels.values[i * num_label_rows + j], training_labels.values[excluded_sample * num_label_rows + j]) =
                    (training_labels.values[excluded_sample * num_label_rows + j], training_labels.values[i * num_label_rows + j]);
                }
            }
        }

        static public Tensor label_to_tensor(Double[,,] label) {
            Tensor temp = new Tensor(2, 1, label.GetLength(2), 1, 1);
            for (int i=0; i < label.GetLength(2); i++) {
                temp.values[i] = label[0, 0, i];
            }
            return temp;
        }
        static public Tensor image_to_tensor(Double[,,] image) {

            int image_x = image.GetLength(0);
            int image_y = image.GetLength(1);
            int image_z = image.GetLength(2);

            Tensor temp = new Tensor(4, 1, image_x, image_y, image_z);
            for (int i=0; i < image_x; i++) {
                for (int j=0; j < image_y; j++) {
                    for (int k=0; k < image_z; k++) {
                        temp.values[i * image_y * image_z + j * image_z + k] = image[i, j, k];
                    }
                }
            }
            return temp;
        }

        static public void printWeightsBiases(Fully_Connected_Layer Inner1, Fully_Connected_Layer Inner2, Fully_Connected_Layer Inner3) {

            Console.WriteLine("Layer 1 weights");
            for (int i = 0; i < Inner1.weights.Count(); i++) {
                for (int j = 0; j < Inner1.weights[0].GetLength(2); j++) {
                    Console.WriteLine(Inner1.weights[i][0, 0, j]);
                }
            }

            Console.WriteLine("Layer 2 weights");
            for (int i = 0; i < Inner2.weights.Count(); i++) {
                for (int j = 0; j < Inner2.weights[0].GetLength(2); j++) {
                    Console.WriteLine(Inner2.weights[i][0, 0, j]);
                }
            }

            Console.WriteLine("Layer 3 weights");
            for (int i = 0; i < Inner3.weights.Count(); i++) {
                for (int j = 0; j < Inner3.weights[0].GetLength(2); j++) {
                    Console.WriteLine(Inner3.weights[i][0, 0, j]);
                }
            }

            Console.WriteLine("Layer 1 biases");
            for (int i = 0; i < Inner1.biases.Count(); i++) {
                Console.WriteLine(Inner1.biases[i][0, 0, 0]);
            }

            Console.WriteLine("Layer 2 biases");
            for (int i = 0; i < Inner2.biases.Count(); i++) {
                Console.WriteLine(Inner2.biases[i][0, 0, 0]);
            }

            Console.WriteLine("Layer 3 biases");
            for (int i = 0; i < Inner3.biases.Count(); i++) {
                Console.WriteLine(Inner3.biases[i][0, 0, 0]);
            }
        }

        static public void printImages(Double[,,] image) {
            int size_x = image.GetLength(0);
            int size_y = image.GetLength(1);
            int size_z = image.GetLength(2);
            string s = "";

            for (int z = 0; z < size_z; z++) {
                for (int x = 0; x < size_x; x++) {
                    for (int y = 0; y < size_y; y++) {
                        if (image[x, y, z] == -1) {
                            s += ".";
                        } else if (image[x, y, z] == 1) {
                            s += "%";
                        } else {
                            s += "o";
                        }
                    }
                    s += "\n";
                }
            }
            Console.WriteLine(s);
        }

        static public void printLabels(Double[,,] label) {
            int size_z = label.GetLength(2);
            string s = "";

            for (int z = 0; z < size_z; z++) {
                s += z;
                s += "\t";
            }
            s += "\n";

            for (int z = 0; z < size_z; z++) {
                s += label[0, 0, z];
                s += "\t";
            }
            s += "\n";
            Console.WriteLine(s);
        }

        static public void printArray(Double[,,] input) {
            int rows = input.GetLength(0);
            int columns = input.GetLength(1);
            int channels = input.GetLength(2);

            Console.WriteLine("Rows: " + rows);
            Console.WriteLine("Columns: " + columns);
            Console.WriteLine("Channels: " + channels);

            Console.Write("(");
            for (int i = 0; i < rows; i++) {
                Console.Write("[");
                for (int j = 0; j < columns; j++) {
                    Console.Write("{");
                    for (int k = 0; k < channels; k++) {
                        Console.Write(input[i, j, k]);
                        if (k < channels - 1) {
                            Console.Write(", ");
                        } else {
                            Console.Write("}");
                        }
                    }
                    if (j < columns - 1) {
                        Console.Write(", ");
                    } else {
                        Console.Write("");
                    }
                }
                Console.Write("]");
                if (i < rows - 1) {
                    Console.WriteLine(",");
                }
                Console.Write("");
            }
            Console.WriteLine(")");
            Console.WriteLine("-----------------------------------------");
        }
    }
}
