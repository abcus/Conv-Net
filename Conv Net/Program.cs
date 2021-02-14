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

        public static Random rand = new Random(0);
        public static MathNet.Numerics.Distributions.Normal normalDist = new MathNet.Numerics.Distributions.Normal(0, 1, rand);
        public static Stopwatch stopwatch = new Stopwatch();

        public static Net NN = new Net();
        public static Double eta = 0.01;
        public static int batchSize = 16;

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            Utils.loadMNIST(60000, 10000, 28, 28, 1, 10);

            ConvolutionLayer conv1 = new ConvolutionLayer(1, 5, 3, 3);
            ReluLayer relu1 = new ReluLayer();
            MaxPoolingLayer pool1 = new MaxPoolingLayer(2, 2, 2);
            FlattenLayer flatten1 = new FlattenLayer();
            FullyConnectedLayer FC1 = new FullyConnectedLayer(3380, 10, true);
            SoftmaxLossLayer softmax = new SoftmaxLossLayer();

            Double[,,] test = { {{12, 1 },{20, 2},{30, 3},{0, 4} },{ {8, 5 },{12, 6 },{2, 7 },{0, 8 } },{ {34, 9 },{70, 10 },{37, 11 },{4, 12 } },{ {112, 13 },{100, 14 },{25, 15},{12, 16 } } };
            Utils.printArray(test);
            test = pool1.forward(test);
            Utils.printArray(test);

            Utils.printArray(pool1.backward(test));

            /*for (int i=0; i < 200; i++) {
                Double[,,] output = trainImageArray[i];
                Double[,,] loss;

                output = conv1.forward(output);
                output = relu1.forward(output);
                output = flatten1.forward(output);
                output = FC1.forward(output);
                output = softmax.forward(output);
                loss = softmax.categoricalCrossEntropyLoss(trainLabelArray[i]);

                Console.WriteLine(i);

                Double[,,] grad;
                grad = softmax.backward();
                grad = FC1.backward(grad);
                grad = flatten1.backward(grad);
                grad = relu1.backward(grad);
                grad = conv1.backward(grad);

                FC1.update(1);
                conv1.update(1);
            }

            int correct = 0;
            for (int i=0; i < 100; i++) {
                Double[,,] output = testImageArray[i + 1000];
                Double[,,] loss;

                output = conv1.forward(output);
                output = flatten1.forward(output);
                output = FC1.forward(output);
                output = softmax.forward(output);
                if (Utils.indexMaxValue(output) == Utils.indexMaxValue(testLabelArray[i + 1000])) {
                    correct++;
                }
                Console.WriteLine(correct + " correct out of " + (i));
            }*/


        
            

            


            /*test();
            for (int epoch = 0; epoch < 10; epoch++) {
                stopwatch.Start();
                Console.WriteLine("++++++++++++++++++++++++++++++++");
                Console.WriteLine("Epoch: " + epoch);
                Utils.shuffleTrainingSet();
                train(batchSize);
                test();
                stopwatch.Stop();
                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed);
                stopwatch.Reset();
            }*/
        }

        static void test() {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;

            for (int i = 0; i < 10000; i++) {
                Tuple<Double[,,], Double[,,]> t;
                t = NN.forward(testImageArray[i], testLabelArray[i]);
                if (Utils.indexMaxValue(t.Item1) == Utils.indexMaxValue(testLabelArray[i])) {
                    correct++;
                }
                totalCrossEntropyLoss += t.Item2[0, 0, 0];
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss);
        }

        static void train(int batchSize) {
            for (int i = 0; i < 60000; i++) {
                Tuple<Double[,,], Double[,,]> t;
                t = NN.forward(trainImageArray[i], trainLabelArray[i]);
                NN.backward();
                if (i % batchSize == 0 || i == 59999) {
                    NN.update(batchSize);
                }
            }
        }
    }
}