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

            Utils.loadMNIST(60000, 10000, 28, 28, 1, 10);
            
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

        static void test () {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;

            for (int testIndex = 0; testIndex < 10000; testIndex++) {
                Tuple <Double[,,], Double[,,]> t;
                t = NN.forward(testImageArray[testIndex], testLabelArray[testIndex]);
                if (Utils.indexMaxValue(t.Item1) == Utils.indexMaxValue(testLabelArray[testIndex])) {
                    correct++;
                }
                totalCrossEntropyLoss += t.Item2[0, 0, 0];
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss);
        }

        static void train () {
            for (int trainIndex = 0; trainIndex < 60000; trainIndex++) {
                Tuple <Double[,,], Double[,,]> t;
                t = NN.forward(trainImageArray[trainIndex], trainLabelArray[trainIndex]);
                NN.backward();
                NN.update();
            }
        }        
    }



    

    

   


    
}

/* To Do:

// 
Image pre-processing
Bath normalization
Regularization (dropout, L2)
Cross valication
Gradient descent (Adam)



convolusion with padding and different stride, test analytic gradient with changes to function, refactor forward function
max pool
Fix loss function (right now it is for binary classification)

// Test analytic gradient with changes to function

*/









