using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using MathNet.Numerics;

namespace Conv_Net {
    static class Program {
        //[STAThread]
        public static Double[][,,] train_image_array;
        public static Double[][,,] train_label_array;
        public static Double[][,,] test_image_array;
        public static Double[][,,] test_label_array;

        public static MathNet.Numerics.Distributions.Normal normalDist = new MathNet.Numerics.Distributions.Normal(0, 1, new Random(0));
        public static Double eta = 0.001;

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            loadMNIST(60000, 10000, 28, 28, 1, 10);

            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss;

            Input_Layer Input = new Input_Layer(28, 28, 1);
            Flatten_Layer Flatten = new Flatten_Layer();
            Inner_Product_Layer Inner_1 = new Inner_Product_Layer(784, 100);
            Relu_Layer Relu_1 = new Relu_Layer();
            Inner_Product_Layer Inner_2 = new Inner_Product_Layer(100, 50);
            Relu_Layer Relu_2 = new Relu_Layer();
            Inner_Product_Layer Inner_3 = new Inner_Product_Layer(50, 10);
            Softmax_Layer Softmax = new Softmax_Layer();
            LossLayer loss = new LossLayer();

            Double[,,] x1 = Input.forward(train_image_array[0]);
            Double[,,] x1_flat = Flatten.forward(x1);
            Double[,,] z1 = Inner_1.forward(x1_flat);
            Double[,,] a1 = Relu_1.forward(z1);
            Double[,,] z2 = Inner_2.forward(a1);
            Double[,,] a2 = Relu_2.forward(z2);
            Double[,,] z3 = Inner_3.forward(a2);
            Double[,,] a3 = Softmax.forward(z3);

            Double[,,] l1 = loss.backward(a3, train_label_array[0]);
            Double[,,] l2 = Softmax.backward(z3);
            Double[,,] l3 = hadamardProduct(l1, l2); // Gradient output layer

            Double[,,] l4 = new Double[1, 1, 50];
            for (int i = 0; i < 50; i++) {
                l4[0, 0, i] = dotProduct(l3, transpose(Inner_3.weights)[i]);
            }
            Double[,,] l5 = Relu_2.backward(z2);
            Double[,,] l6 = hadamardProduct(l4, l5); // Gradient hidden layer 2


            Double[,,] l7 = new Double[1, 1, 100];
            for (int i = 0; i < 100; i++) {
                l7[0, 0, i] = dotProduct(l6, transpose(Inner_2.weights)[i]);
            }
            Double[,,] l8 = Relu_1.backward(z1);
            Double[,,] l9 = hadamardProduct(l7, l8); // Gradient hidden layer 1




            Inner_3.weights = updateWeights(Inner_3.weights, l3, a2);
            Inner_3.biases = updateBiases(Inner_3.biases, l3);
            Inner_2.weights = updateWeights(Inner_2.weights, l6, a1);
            Inner_2.biases = updateBiases(Inner_2.biases, l6);
            Inner_1.weights = updateWeights(Inner_1.weights, l9, x1_flat);
            Inner_1.biases = updateBiases(Inner_1.biases, l9);

            /*for (int i = 0; i < 10000; i++) {
                Double[,,] x1 = Input.forward(test_image_array[i]);
                Double[,,] x1_flat = Flatten.forward(x1);
                Double[,,] z1 = Inner_1.forward(x1_flat);
                Double[,,] a1 = Relu_1.forward(z1);
                Double[,,] z2 = Inner_2.forward(a1);
                Double[,,] a2 = Relu_2.forward(z2);
                Double[,,] z3 = Inner_3.forward(a2);
                Double[,,] a3 = Softmax.forward(z3);
                Double[,,] l1 = loss.forward(a3, test_label_array[i]);
                totalCrossEntropyLoss += l1[0, 0, 0];
                if (indexMaxValue(a3) == indexMaxValue(test_label_array[i])) {
                    correct++;
                }
            }
            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + totalCrossEntropyLoss / 10000);*/




        }

        static void loadMNIST(int num_train, int num_test, int input_size_x, int input_size_y, int input_size_z, int label_size_z) {
            Double[][,,] temp_train_image_array = new Double[num_train][,,];
            Double[][,,] temp_train_label_array = new Double[num_train][,,];

            Double[][,,] temp_test_image_array = new Double[num_test][,,];
            Double[][,,] temp_test_label_array = new Double[num_test][,,];

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
                for (int image_num = 0; image_num < num_train; image_num++) {

                    Double[,,] temp_image = new Double[input_size_x, input_size_y, input_size_z];

                    // Load image
                    for (int image_pos_x = 0; image_pos_x < input_size_x; image_pos_x++) {
                        for (int image_pos_y = 0; image_pos_y < input_size_y; image_pos_y++) {
                            for (int image_pos_z = 0; image_pos_z < input_size_z; image_pos_z++) {
                                Double b = brTrainImages.ReadByte();
                                b = (-1 + (b / 127.5));
                                temp_image[image_pos_x, image_pos_y, image_pos_z] = b;
                                temp_train_image_array[image_num] = temp_image;
                            }
                        }
                    }
                    // Load label
                    int label = brTrainLabels.ReadByte();
                    Double[,,] temp_label = new Double[1, 1, label_size_z];
                    temp_label[0, 0, label] = 1.0;
                    temp_train_label_array[image_num] = temp_label;
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
                for (int image_num = 0; image_num < num_test; image_num++) {

                    Double[,,] temp_image = new Double[input_size_x, input_size_y, input_size_z];

                    // Load image
                    for (int image_pos_x = 0; image_pos_x < input_size_x; image_pos_x++) {
                        for (int image_pos_y = 0; image_pos_y < input_size_y; image_pos_y++) {
                            for (int image_pos_z = 0; image_pos_z < input_size_z; image_pos_z++) {
                                Double b = brTestImages.ReadByte();
                                b = (-1 + (b / 127.5));
                                temp_image[image_pos_x, image_pos_y, image_pos_z] = b;
                                temp_test_image_array[image_num] = temp_image;
                            }
                        }
                    }
                    // Load label
                    int label = brTestLabels.ReadByte();
                    Double[,,] temp_label = new Double[1, 1, label_size_z];
                    temp_label[0, 0, label] = 1.0;
                    temp_test_label_array[image_num] = temp_label;
                }
                testImagesStream.Close();
                brTestImages.Close();

                testLabelsStream.Close();
                brTestLabels.Close();

                train_image_array = temp_train_image_array;
                train_label_array = temp_train_label_array;
                test_image_array = temp_test_image_array;
                test_label_array = temp_test_label_array;
            } catch {

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

        static public Double dotProduct(Double[,,] x, Double[,,] y) {
            Debug.Assert(x.GetLength(2) == y.GetLength(2));
            Double dotProduct = 0.0;
            for (int i = 0; i < x.GetLength(2); i++) {
                dotProduct += (x[0, 0, i] * y[0, 0, i]);
            }
            return dotProduct;
        }

        static void printArray(Double[,,] input) {
            for (int i = 0; i < input.GetLength(2); i++) {
                Console.WriteLine(input[0, 0, i]);
            }
        }

        static public Double[,,] hadamardProduct(Double[,,] x, Double[,,] y) {
            Debug.Assert(x.GetLength(2) == y.GetLength(2));
            int size = x.GetLength(2);
            Double[,,] output = new Double[1, 1, size];
            for (int i = 0; i < size; i++) {
                output[0, 0, i] = x[0, 0, i] * y[0, 0, i];
            }
            return output;
        }


        static int indexMaxValue(Double[,,] input) {
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

        static Double maxValue(Double[,,] input) {
            Double max = Double.MinValue;
            for (int i = 0; i < input.GetLength(2); i++) {
                if (input[0, 0, i] > max) {
                    max = input[0, 0, i];
                }
            }
            return max;
        }

        static Double[][,,] transpose(Double[][,,] input) {
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

        static public Double[][,,] updateBiases(Double[][,,] biases, Double[,,] gradients) {
            Debug.Assert(biases.Count() == gradients.GetLength(2));

            Double[][,,] output = new Double[biases.Count()][,,];
            for (int i=0; i < biases.Count(); i++) {
                output[i] = new Double[1, 1, 1];
                output[i][0, 0, 0] = biases[i][0, 0, 0] - gradients[0, 0, i] * 1 * eta;
            }
            return output;
        }

        static public Double[][,,] updateWeights(Double[][,,] weights, Double[,,] gradients, Double[,,] outputsPreviousLayer) {
            Debug.Assert(weights.Count() == gradients.GetLength(2));
            
            Double[][,,] output = new Double[weights.Count()][,,];
            for (int i=0; i < weights.Count(); i++) {
                output[i] = new Double[1, 1, weights[0].GetLength(2)];
                for (int j=0; j < weights[0].GetLength(2); j++) {
                    output[i][0, 0, j] = weights[i][0, 0, j] - gradients[0, 0, i] * outputsPreviousLayer[0,0,j] * eta;
                }
            }
            return output;
        }


        static void test() {
           
        }
    }



    class Input_Layer {

        private int input_size_x;
        private int input_size_y;
        private int input_size_z;

        public Input_Layer (int input_size_x, int input_size_y, int input_size_z) {
            this.input_size_x = input_size_x;
            this.input_size_y = input_size_y;
            this.input_size_z = input_size_z;
        }

        public Double[,,] forward(Double[,,] input) {
            Double[,,] output = new Double[input_size_x, input_size_y, input_size_z];
            for (int i=0; i < input_size_x; i++) {
                for (int j=0; j < input_size_y; j++) {
                    for (int k=0; k < input_size_z; k++) {
                        output[i, j, k] = input[i, j, k];
                    }
                }
            }
            return output;
        }

    }

    class Flatten_Layer {

        public Flatten_Layer () {

        }

        public Double[,,] forward (Double [,,] input) {
            int input_size_x = input.GetLength(0);
            int input_size_y = input.GetLength(1);
            int input_size_z = input.GetLength(2);
            int output_size_z = input_size_x * input_size_y * input_size_z;
            Double[,,] output = new Double[1, 1, output_size_z];
            
            for (int input_pos_x = 0; input_pos_x < input_size_x; input_pos_x ++) {
                for (int input_pos_y = 0; input_pos_y < input_size_y; input_pos_y++) {
                    for (int input_pos_z = 0; input_pos_z < input_size_z; input_pos_z++) {
                        output[0, 0, input_pos_x * input_size_y * input_size_z + input_pos_y * input_size_z + input_pos_z] = input[input_pos_x, input_pos_y, input_pos_z];
                    }
                }
            }
            return output;
        }
    }

    class Inner_Product_Layer {

        private int previous_layer_size;
        private int layer_size;
        public Double[][,,] weights;
        public Double[][,,] biases;
        private Double[][,,] weightGradients;
        private Double[][,,] biasGradients;

        public Inner_Product_Layer(int previous_layer_size, int layer_size) {
            this.previous_layer_size = previous_layer_size;
            this.layer_size = layer_size;
            this.weights = new Double[layer_size][,,];
            this.biases = new Double[layer_size][,,];
            this.weightGradients = new Double[layer_size][,,];
            this.biasGradients = new Double[layer_size][,,];

            for (int i = 0; i < layer_size; i++) {
                Double[,,] temp_weights = new Double[1, 1, previous_layer_size];

                for (int j = 0; j < previous_layer_size; j++) {
                    temp_weights[0,0,j] = Program.normalDist.Sample() * Math.Sqrt(2 / (Double)previous_layer_size);
                }
                this.weights[i] = temp_weights;
            }

            for (int i=0; i < layer_size; i++) {
                Double[,,] temp_biases = new Double[1, 1, 1];
                temp_biases[0, 0, 0] = 0.0;
                this.biases[i] = temp_biases;
            }

            for (int i=0; i < layer_size; i++) {
                Double[,,] tempWeightGradient = new Double[1, 1, previous_layer_size];
                this.weightGradients[i] = tempWeightGradient;
                
                Double[,,] tempBiasGradient = new Double[1, 1, 1];
                this.biasGradients[i] = tempBiasGradient;
            }
        }

        public Double[,,] forward(Double[,,] input) {
            Double[,,] output = new Double[1, 1, layer_size];
            for (int i=0; i < layer_size; i ++) {

                output[0, 0, i] = Program.dotProduct(input, weights[i]) + biases[0][0, 0, 0];
            }
            return output;
        }
    }

    class Relu_Layer {

        public Relu_Layer () {
        
        }

        public Double[,,] forward(Double[,,] input) {
            int input_size_x = input.GetLength(0);
            int input_size_y = input.GetLength(1);
            int input_size_z = input.GetLength(2);
            Double[,,] output = new Double[input_size_x, input_size_y, input_size_z];

            for (int i=0; i < input_size_x; i++) {
                for (int j=0; j < input_size_y; j++) {
                    for (int k=0; k < input_size_z; k++) {
                        output[i, j, k] = input[i, j, k] >= 0 ? input[i, j, k] : 0;
                    }
                }
            }
            return output;
        }

        public Double[,,] backward(Double[,,] input) {
            int input_size_x = input.GetLength(0);
            int input_size_y = input.GetLength(1);
            int input_size_z = input.GetLength(2);
            Double[,,] output = new Double[input_size_x, input_size_y, input_size_z];

            for (int i = 0; i < input_size_x; i++) {
                for (int j = 0; j < input_size_y; j++) {
                    for (int k = 0; k < input_size_z; k++) {
                        output[i, j, k] = input[i, j, k] >= 0 ? 1 : 0;
                    }
                }
            }
            return output;
        }
    }

    class Softmax_Layer {

        public Softmax_Layer () {
        }
        public Double[,,] forward (Double [,,] input) {

            int layer_size = input.GetLength(2);

            Double[,,] output = new Double[1, 1, layer_size];

            // Find max value of input array
            Double max = Double.MinValue;
            for (int i=0; i < layer_size; i ++) {
                if (input[0,0,i] > max) {
                    max = input[0, 0, i];
                }
            }
            
            // Subtract max value of input array from all values
            for (int i=0; i < layer_size; i++) {
                input[0, 0, i] -= max;
            }
            
            // Calculate denominator of softmax
            Double denominator = 0.0;
            for (int i=0; i < layer_size; i ++) {
                denominator += Math.Exp(input[0, 0, i]);
            }

            // Set output array
            for (int i=0; i < layer_size; i++) {
                output[0, 0, i] = Math.Exp(input[0, 0, i]) / denominator;
            }

            return output;
        }

        public Double[,,] backward (Double [,,] input) {
            Double numerator = 0.0;
            Double denominator = 0.0;
            int layerSize = input.GetLength(2);
            Double[,,] output = new Double[1, 1, layerSize];

            for (int i = 0; i < layerSize; i++) {
                denominator += Math.Exp(input[0, 0, i]);
            }
            denominator = Math.Pow(denominator, 2);

            

            for (int i=0; i < layerSize; i++) {
                numerator = 0.0;
                for (int j=0; j < layerSize; j++) {
                    if (j != i) {
                        numerator += Math.Exp(input[0, 0, j]);
                    }
                }
                numerator *= Math.Exp(input[0, 0, i]);
                output[0, 0, i] = numerator / denominator;
            }
            return output;
        }
    }

    class LossLayer {

        public LossLayer () {

        }

        public Double[,,] forward (Double [,,] input, Double[,,] target) {
            int layerSize = input.GetLength(2);
            Double[,,] output = new Double[1, 1, 1];
            Double error = 0.0;

            for (int i=0; i < layerSize; i ++) {
                error += (target[0, 0, i] * Math.Log(input[0, 0, i]) + (1 - target[0, 0, i]) * Math.Log(1 - input[0, 0, i]));
            }
            error *= -1 / (Double)layerSize;
            output[0, 0, 0] = error;
            return output;
        }

        public Double[,,] backward (Double[,,] input, Double[,,] target) {
            int layerSize = input.GetLength(2);
            Double[,,] output = new Double[1, 1, layerSize];
            for (int i=0; i < layerSize; i++) {
                output[0, 0, i] = (- target[0,0,i] + input[0,0,i]) / (input[0,0,i] * (1 - input[0,0,i]));
            }
            return output;
        }

    }
















    class Convolution_Layer {

        private int num_filters;
        private int filter_size;
        private int stride;
        private Double[,,,] filter;
        private Double[] biases;
        

        public Convolution_Layer(int input_z, int num_filters, int filter_size, int stride) {

            this.num_filters = num_filters;
            this.filter_size = filter_size;
            this.stride = stride;

            filter = new Double[num_filters, filter_size, filter_size, input_z];
            // Initialize filter weights

            biases = new Double[num_filters];
        }

        private Double[,,] forward (Double[,,] input) {
            int input_x = input.GetLength(0);
            int input_y = input.GetLength(1);
            int input_z = input.GetLength(2);

            int output_x = (input_x - filter_size) / stride + 1;
            int output_y = (input_y - filter_size) / stride + 1;
            int output_z = num_filters;

            Double[,,] output = new Double[output_x, output_y, output_z];

            Double dot_product = 0.0;

            for (int filter_index = 0; filter_index < num_filters; filter_index++) {
                for (int input_x_pos = 0; input_x_pos <= input_x - filter_size; input_x_pos += stride) {
                    for (int input_y_pos = 0; input_y_pos <= input_y - filter_size; input_y_pos += stride) {
                        for (int filter_x_pos = 0; filter_x_pos < filter_size; filter_x_pos++) {
                            for (int filter_y_pos = 0; filter_y_pos < filter_size; filter_y_pos++) {
                                for (int filter_z_pos = 0; filter_z_pos < input_z; filter_z_pos++) {
                                    dot_product += filter[filter_index, filter_x_pos, filter_y_pos, filter_z_pos] * input[input_x_pos, input_y_pos, filter_z_pos];
                                }
                            }
                        }
                        dot_product += biases[filter_index];
                        output[input_x_pos / stride, input_y_pos / stride, filter_index] = dot_product;
                        dot_product = 0.0;
                    }
                }
            }
            return output;
        }

    }
}

/* To Do:
Padding
ADAM
Regularization (dropout, L2)
Batch normalization
 */


// Forward convolusion











