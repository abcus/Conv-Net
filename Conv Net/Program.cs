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
        public static Double[][,,] trainImageArray;
        public static Double[][,,] trainLabelArray;
        public static Double[][,,] testImageArray;
        public static Double[][,,] testLabelArray;

        public static MathNet.Numerics.Distributions.Normal normalDist = new MathNet.Numerics.Distributions.Normal(0, 1, new Random(0));
        public static Double eta = 0.001;

        public static int layer1size = 20;
        public static int layer2size = 16;
        public static int layer3size = 10;


        public static InputLayer Input = new InputLayer(28, 28, 1);
        public static FlattenLayer Flatten = new FlattenLayer();
        public static InnerProductLayer Inner1 = new InnerProductLayer(784, layer1size);
        public static ReluLayer Relu1 = new ReluLayer();
        public static InnerProductLayer Inner2 = new InnerProductLayer(layer1size, layer2size);
        public static ReluLayer Relu2 = new ReluLayer();
        public static InnerProductLayer Inner3 = new InnerProductLayer(layer2size, layer3size);
        public static SoftmaxLayer Softmax = new SoftmaxLayer();
        public static LossLayer Loss = new LossLayer();

        public static Double[,,] x1, x1_flat, z1, a1, z2, a2, z3, a3;
        public static Double[,,] ltot, l1, l2, l3, l4, l5, l6, l7, l8, l9;

        public static Stopwatch stopwatch = new Stopwatch();

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            loadMNIST(60000, 10000, 28, 28, 1, 10);




            test();
            for (int epoch=0; epoch < 10; epoch++) {
                stopwatch.Start();
                Console.WriteLine("++++++++++++++++++++++++++++++++");
                Console.WriteLine("Epoch: " + epoch);
                train();
                test();
                stopwatch.Stop();
                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed);
                stopwatch.Reset();
            }
        }

        static void loadMNIST(int numTrain, int numTest, int inputSizeX, int inputSizeY, int inputSizeZ, int labelSize) {
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

                trainImageArray = tempTrainImageArray;
                trainLabelArray = tempTrainLabelArray;
                testImageArray = tempTestImageArray;
                testLabelArray = tempTestLabelArray;
            } catch {

            }
        }

        static void test () {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;

            for (int testIndex = 0; testIndex < 10000; testIndex++) {
                x1 = Input.forward(testImageArray[testIndex]);
                x1_flat = Flatten.forward(x1);
                z1 = Inner1.forward(x1_flat);
                a1 = Relu1.forward(z1);
                z2 = Inner2.forward(a1);
                a2 = Relu2.forward(z2);
                z3 = Inner3.forward(a2);
                a3 = Softmax.forward(z3);
                ltot = Loss.forward(a3, testLabelArray[testIndex]);
                totalCrossEntropyLoss += ltot[0, 0, 0];
                if (indexMaxValue(a3) == indexMaxValue(testLabelArray[testIndex])) {
                    correct++;
                }
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss);
        }

        static void train () {
            for (int trainIndex = 0; trainIndex < 60000; trainIndex++) {
                x1 = Input.forward(trainImageArray[trainIndex]);
                x1_flat = Flatten.forward(x1);
                z1 = Inner1.forward(x1_flat);
                a1 = Relu1.forward(z1);
                z2 = Inner2.forward(a1);
                a2 = Relu2.forward(z2);
                z3 = Inner3.forward(a2);
                a3 = Softmax.forward(z3);
                
                l1 = Loss.backward(a3, trainLabelArray[trainIndex]);
                l2 = Softmax.backward(z3);
                l3 = hadamardProduct(l1, l2); // Gradient output layer

                l4 = new Double[1, 1, layer2size];
                for (int i = 0; i < layer2size; i++) {
                    l4[0, 0, i] = dotProduct(l3, transpose(Inner3.weights)[i]);
                }
                l5 = Relu2.backward(z2);
                l6 = hadamardProduct(l4, l5); // Gradient hidden layer 2

                l7 = new Double[1, 1, layer1size];
                for (int i = 0; i < layer1size; i++) {
                    l7[0, 0, i] = dotProduct(l6, transpose(Inner2.weights)[i]);
                }
                l8 = Relu1.backward(z1);
                l9 = hadamardProduct(l7, l8); // Gradient hidden layer 1

                Inner3.weights = updateWeights(Inner3.weights, l3, a2);
                Inner3.biases = updateBiases(Inner3.biases, l3);
                Inner2.weights = updateWeights(Inner2.weights, l6, a1);
                Inner2.biases = updateBiases(Inner2.biases, l6);
                Inner1.weights = updateWeights(Inner1.weights, l9, x1_flat);
                Inner1.biases = updateBiases(Inner1.biases, l9);
            }
        }

        static void printWeightsBiases () {

            Console.WriteLine("Layer 1 weights");
            for (int i=0; i < Inner1.weights.Count(); i++) {
                for (int j=0; j < Inner1.weights[0].GetLength(2); j++) {
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



    }



    class InputLayer {

        private int input_size_x;
        private int input_size_y;
        private int input_size_z;

        public InputLayer (int input_size_x, int input_size_y, int input_size_z) {
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

    class FlattenLayer {

        public FlattenLayer () {

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

    class InnerProductLayer {

        private int previous_layer_size;
        private int layer_size;
        public Double[][,,] weights;
        public Double[][,,] biases;
        private Double[][,,] weightGradients;
        private Double[][,,] biasGradients;

        public InnerProductLayer(int previous_layer_size, int layer_size) {
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

                output[0, 0, i] = Program.dotProduct(input, weights[i]) + biases[i][0, 0, 0];
            }
            return output;
        }
    }

    class ReluLayer {

        public ReluLayer () {
        
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

    class SoftmaxLayer {

        public SoftmaxLayer () {
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











