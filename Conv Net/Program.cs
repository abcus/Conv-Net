﻿using System;
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
        public static ConvNet CNN = new ConvNet();
        public static Double eta = 0.01;
        public static int batchSize = 16;

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            Utils.loadMNIST(60000, 10000, 28, 28, 1, 10);

            testCNN();
            for (int epoch = 0; epoch < 10; epoch ++) {
                Console.WriteLine("------------------------------------------");
                Console.WriteLine("Epoch: " + epoch);
                Utils.shuffleTrainingSet();

                stopwatch.Start();
                trainCNN(batchSize);
                stopwatch.Stop();
                Console.WriteLine("Time elapsed for training: " + stopwatch.Elapsed);
                stopwatch.Reset();
                
                testCNN();
                
            }


            /*testNN();
            for (int epoch = 0; epoch < 10; epoch++) {
                stopwatch.Start();
                Console.WriteLine("++++++++++++++++++++++++++++++++");
                Console.WriteLine("Epoch: " + epoch);
                Utils.shuffleTrainingSet();
                trainNN(batchSize);
                testNN();
                stopwatch.Stop();
                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed);
                stopwatch.Reset();
            }*/
        }

        static void testCNN() {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;

            for (int i = 0; i < 10000; i++) {
                Tuple<Double[,,], Double[,,]> t;
                t = CNN.forward(testImageArray[i], testLabelArray[i]);
                if (Utils.indexMaxValue(t.Item1) == Utils.indexMaxValue(testLabelArray[i])) {
                    correct++;
                }
                totalCrossEntropyLoss += t.Item2[0, 0, 0];
            }
            averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

            Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss + "\n\n");
        }

        static void trainCNN(int batchSize) {
            for (int i = 0; i < 60000; i++) {
                Tuple<Double[,,], Double[,,]> t;
                t = CNN.forward(trainImageArray[i], trainLabelArray[i]);
                CNN.backward();
                if (i % batchSize == 0 || i == 59999) {
                    CNN.update(batchSize);
                }
            }
        }

        static void testNN() {
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

        static void trainNN(int batchSize) {
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