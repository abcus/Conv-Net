using System;
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


        public static void loadMNIST(int numTrain, int numTest, int inputSizeX, int inputSizeY, int inputSizeZ, int labelSize) {
            Double[][,,] tempTrainImageArray = new Double[numTrain][,,];
            Double[][,,] tempTrainLabelArray = new Double[numTrain][,,];

            Double[][,,] tempTestImageArray = new Double[numTest][,,];
            Double[][,,] tempTestLabelArray = new Double[numTest][,,];

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
                for (int image_num = 0; image_num < numTrain; image_num++) {

                    Double[,,] temp_image = new Double[inputSizeX, inputSizeY, inputSizeZ];

                    // Load image
                    for (int image_pos_x = 0; image_pos_x < inputSizeX; image_pos_x++) {
                        for (int image_pos_y = 0; image_pos_y < inputSizeY; image_pos_y++) {
                            for (int image_pos_z = 0; image_pos_z < inputSizeZ; image_pos_z++) {
                                Double b = brTrainImages.ReadByte();
                                b = (-1 + (b / 127.5));
                                temp_image[image_pos_x, image_pos_y, image_pos_z] = b;
                                tempTrainImageArray[image_num] = temp_image;
                            }
                        }
                    }
                    // Load label
                    int label = brTrainLabels.ReadByte();
                    Double[,,] temp_label = new Double[1, 1, labelSize];
                    temp_label[0, 0, label] = 1.0;
                    tempTrainLabelArray[image_num] = temp_label;
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
                for (int image_num = 0; image_num < numTest; image_num++) {

                    Double[,,] temp_image = new Double[inputSizeX, inputSizeY, inputSizeZ];

                    // Load image
                    for (int image_pos_x = 0; image_pos_x < inputSizeX; image_pos_x++) {
                        for (int image_pos_y = 0; image_pos_y < inputSizeY; image_pos_y++) {
                            for (int image_pos_z = 0; image_pos_z < inputSizeZ; image_pos_z++) {
                                Double b = brTestImages.ReadByte();
                                b = (-1 + (b / 127.5));
                                temp_image[image_pos_x, image_pos_y, image_pos_z] = b;
                                tempTestImageArray[image_num] = temp_image;
                            }
                        }
                    }
                    // Load label
                    int label = brTestLabels.ReadByte();
                    Double[,,] temp_label = new Double[1, 1, labelSize];
                    temp_label[0, 0, label] = 1.0;
                    tempTestLabelArray[image_num] = temp_label;
                }
                testImagesStream.Close();
                brTestImages.Close();

                testLabelsStream.Close();
                brTestLabels.Close();

                Program.trainImageArray = tempTrainImageArray;
                Program.trainLabelArray = tempTrainLabelArray;
                Program.testImageArray = tempTestImageArray;
                Program.testLabelArray = tempTestLabelArray;
            } catch {

            }
        }

        static void printWeightsBiases(FullyConnectedLayer Inner1, FullyConnectedLayer Inner2, FullyConnectedLayer Inner3) {

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

        static void printImages(Double[,,] image) {
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

        static void printLabels(Double[,,] label) {
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
            int x = input.GetLength(0);
            int y = input.GetLength(1);
            int z = input.GetLength(2);

            Console.WriteLine("");
            Console.WriteLine("x: " + x);
            Console.WriteLine("y: " + y);
            Console.WriteLine("z: " + z);
            Console.WriteLine("");

            Console.Write("(");
            for (int i = 0; i < x; i++) {
                Console.Write("[");
                for (int j = 0; j < y; j++) {
                    Console.Write("{");
                    for (int k = 0; k < z; k++) {
                        Console.Write(input[i, j, k]);
                        if (k < z - 1) {
                            Console.Write(", ");
                        } else {
                            Console.Write("}");
                        }
                    }
                    if (j < y - 1) {
                        Console.WriteLine(", ");
                    } else {
                        Console.Write("");
                    }
                }

                Console.Write("]");
                if (i < x - 1) {
                    Console.WriteLine(",");
                }
                Console.Write("");
            }
            Console.Write(")");
        }
    }
}
