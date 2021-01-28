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
        public static Stopwatch stopwatch = new Stopwatch();

        public static Net NN = new Net();
        public static Double eta = 0.001;

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            /*Utils.loadMNIST(60000, 10000, 28, 28, 1, 10);
            
            
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
            }*/

            Double[,,] test = { { {1,10 },{2,11 },{3,12 } },{ {4,13 },{5,14 },{6,15 } },{ {7,16 },{8,17 },{9,18 } } };
            Double[,,] filter = { { {2,6 },{3,7 } }, { {4,8 },{5,9 } } };
            Double[,,] filter2 = { { {4, 4 },{7, 5 } },{ {8, 7 },{9, 9 } } };
            Utils.printArray(test);
            Utils.printArray(filter);
            ConvolutionLayer conv = new ConvolutionLayer(2, 2, 2, 1, filter, filter2);
            Utils.printArray(conv.forward(test));
        }

        static void test () {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;

            for (int testIndex = 0; testIndex < 10000; testIndex++) {
                Double[,,] t;
                t = NN.forward(testImageArray[testIndex]);
                if (Utils.indexMaxValue(t) == Utils.indexMaxValue(testLabelArray[testIndex])) {
                    correct++;
                }
                t = NN.loss(t, testLabelArray[testIndex]);
                totalCrossEntropyLoss += t[0, 0, 0];
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss);
        }

        static void train () {
            for (int trainIndex = 0; trainIndex < 60000; trainIndex++) {

                Double[,,] t;
                t = NN.forward(trainImageArray[trainIndex]);
                t = NN.loss(t, trainLabelArray[trainIndex]);
                NN.backward();
                NN.update();
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









