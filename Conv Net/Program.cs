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

        //public static Net NN = new Net();

        public static InputLayer Input = new InputLayer(28, 28, 1);
        public static FlattenLayer Flatten = new FlattenLayer();
        public static FullyConnectedLayer FC1 = new FullyConnectedLayer(784, 5);
        public static ReluLayer Relu1 = new ReluLayer();
        public static FullyConnectedLayer FC2 = new FullyConnectedLayer(5, 6);
        public static ReluLayer Relu2 = new ReluLayer();
        public static FullyConnectedLayer FC3 = new FullyConnectedLayer(6, 10);
        public static SoftmaxLayer Softmax = new SoftmaxLayer();
        public static LossLayer Loss = new LossLayer();

        public static Stopwatch stopwatch = new Stopwatch();

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            Utils.loadMNIST(60000, 10000, 28, 28, 1, 10);
            
            
            /*Double totalCrossEntropyLosss = 0.0;
            int correctt = 0;

            for (int i=0; i < 10000; i++) {
                
                Double [,,] result = NN.forward(testImageArray[i]);
                Double[,,] loss = NN.loss(result, testLabelArray[i]);
                totalCrossEntropyLosss += loss[0, 0, 0];
                if (Utils.indexMaxValue(result) == Utils.indexMaxValue(testLabelArray[i])) {
                    correctt++;
                }

            }
            Console.WriteLine(totalCrossEntropyLosss / 10000);
            Console.WriteLine(correctt);

            for (int i=0; i < 60000; i ++) {
                Double[,,] result = NN.forward(trainImageArray[i]);
                Double[,,] loss = NN.loss(result, trainLabelArray[i]);
                NN.backward();
                NN.update();
            }

            for (int i = 0; i < 10000; i++) {

                Double[,,] result = NN.forward(testImageArray[i]);
                Double[,,] loss = NN.loss(result, testLabelArray[i]);
                totalCrossEntropyLosss += loss[0, 0, 0];
                if (Utils.indexMaxValue(result) == Utils.indexMaxValue(testLabelArray[i])) {
                    correctt++;
                }

            }
            Console.WriteLine(totalCrossEntropyLosss / 10000);
            Console.WriteLine(correctt);*/

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


        static Double[,,] forward(Double[,,] input) {
            Double[,,] t;
            t = Input.forward(input);
            t = Flatten.forward(t);
            t = FC1.forward(t);
            t = Relu1.forward(t);
            t = FC2.forward(t);
            t = Relu2.forward(t);
            t = FC3.forward(t);
            t = Softmax.forward(t);
            return t;
        }
        static Double[,,] loss(Double[,,] input, Double[,,] target) {
            return Loss.forward(input, target);
        }

        static void backward () {
            Double[,,] l;
            // for output layer
            l = Loss.backward();
            l = Softmax.backward(l);
            l = FC3.backward(l);

            // Hidden layer 2
            l = Relu2.backward(l);
            l = FC2.backward(l);

            // Hidden layer 1
            l = Relu1.backward(l);
            FC1.storeGradient(l);
        }

        static void update() {
            FC3.update();
            FC2.update();
            FC1.update();
        }

        static void test () {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;

            for (int testIndex = 0; testIndex < 10000; testIndex++) {
                Double[,,] t;
                t = forward(testImageArray[testIndex]);
                if (Utils.indexMaxValue(t) == Utils.indexMaxValue(testLabelArray[testIndex])) {
                    correct++;
                }
                t = loss(t, testLabelArray[testIndex]);
                totalCrossEntropyLoss += t[0, 0, 0];
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss);
        }

        static void train () {
            for (int trainIndex = 0; trainIndex < 60000; trainIndex++) {

                Double[,,] t;
                t = forward(trainImageArray[trainIndex]);
                t = loss(t, trainLabelArray[trainIndex]);
                backward();
                update();
            }
        }        
    }



    

    

   


    
}

/* To Do:

ADAM
Regularization (dropout, L2)
Batch normalization

convolusion with padding
max pool

*/









