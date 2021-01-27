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

        public static int layer1size = 5;
        public static int layer2size = 6;
        public static int layer3size = 10;

        public static InputLayer Input = new InputLayer(28, 28, 1);
        public static FlattenLayer Flatten = new FlattenLayer();
        public static FullyConnectedLayer Inner1 = new FullyConnectedLayer(784, layer1size);
        public static ReluLayer Relu1 = new ReluLayer();
        public static FullyConnectedLayer Inner2 = new FullyConnectedLayer(layer1size, layer2size);
        public static ReluLayer Relu2 = new ReluLayer();
        public static FullyConnectedLayer Inner3 = new FullyConnectedLayer(layer2size, layer3size);
        public static SoftmaxLayer Softmax = new SoftmaxLayer();
        public static LossLayer Loss = new LossLayer();

        public static Double[,,] x0, x0Flat, z1, a1, z2, a2, z3, a3, c4;
        public static Double[,,] ltot, d1, d2, d3, d4, d5, d6;

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
                x0 = Input.forward(testImageArray[testIndex]);
                x0Flat = Flatten.forward(x0);
                z1 = Inner1.forward(x0Flat);
                a1 = Relu1.forward(z1);
                z2 = Inner2.forward(a1);
                a2 = Relu2.forward(z2);
                z3 = Inner3.forward(a2);
                a3 = Softmax.forward(z3);
                ltot = Loss.forward(a3, testLabelArray[testIndex]);
                totalCrossEntropyLoss += ltot[0, 0, 0];
                if (Utils.indexMaxValue(a3) == Utils.indexMaxValue(testLabelArray[testIndex])) {
                    correct++;
                }
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss);
        }

        static void train () {
            for (int trainIndex = 0; trainIndex < 60000; trainIndex++) {
                
                // input layer
                x0 = Input.forward(trainImageArray[trainIndex]);
                x0Flat = Flatten.forward(x0);
                
                // Hidden layer 1
                z1 = Inner1.forward(x0Flat);
                a1 = Relu1.forward(z1);
                
                // Hidden layer 2
                z2 = Inner2.forward(a1);
                a2 = Relu2.forward(z2);
                
                // Hidden layer 3
                z3 = Inner3.forward(a2);
                a3 = Softmax.forward(z3);
                c4 = Loss.forward(a3, trainLabelArray[trainIndex]);



                // for output layer
                d1 = Loss.backward();   
                d2 = Softmax.backward(d1); 
                d3 = Inner3.backward(d2);

                // Hidden layer 2
                d4 = Relu2.backward(d3);              
                d5 = Inner2.backward(d4);

                // Hidden layer 1
                d6 = Relu1.backward(d5); 

                Inner3.weights = updateWeights(Inner3.weights, d2, a2);
                Inner3.biases = updateBiases(Inner3.biases, d2);
                Inner2.weights = updateWeights(Inner2.weights, d4, a1);
                Inner2.biases = updateBiases(Inner2.biases, d4);
                Inner1.weights = updateWeights(Inner1.weights, d6, x0Flat);
                Inner1.biases = updateBiases(Inner1.biases, d6);
            }
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



    

    

   


    
}

/* To Do:
Padding
ADAM
Regularization (dropout, L2)
Batch normalization
 */


// Forward convolusion











