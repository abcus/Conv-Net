using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

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

        static void printArray(Double[,,] input) {
            for (int i = 0; i < input.GetLength(2); i++) {
                Console.WriteLine(input[0, 0, i]);
            }
        }
    }
}
