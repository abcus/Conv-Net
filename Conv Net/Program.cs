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

        public static Tensor training_images;
        public static Tensor training_labels;
        public static Tensor testing_images;
        public static Tensor testing_labels;

        public static Random rand = new Random(0);
        public static MathNet.Numerics.Distributions.Normal normalDist = new MathNet.Numerics.Distributions.Normal(0, 1, rand);
        public static Stopwatch stopwatch = new Stopwatch();

        public static Net NN = new Net();
        //public static ConvNet CNN = new ConvNet();
        public static Double eta = 0.01;
        public static int batchSize = 16;

        public static int testing_sample_size = 10000;

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            Tuple<Tensor, Tensor, Tensor, Tensor> data = Utils.load_MNIST(60000, 10000, 28, 28, 1, 10);
            training_images = data.Item1;
            training_labels = data.Item2;
            testing_images = data.Item3;
            testing_labels = data.Item4;
            
            testImageArray = testing_images.convert_to_array();
            testLabelArray = testing_labels.convert_to_array();

            //testCNN(testing_sample_size);
            //for (int epoch = 0; epoch < 10; epoch++) {
            //    Console.WriteLine("------------------------------------------");
            //    Console.WriteLine("Epoch: " + epoch);

            //    Utils.shuffle_Tensor(training_images, training_labels);
            //    trainImageArray = training_images.convert_to_array();
            //    trainLabelArray = training_labels.convert_to_array();

            //    stopwatch.Start();
            //    trainCNN(batchSize);
            //    stopwatch.Stop();
            //    Console.WriteLine("Time elapsed for training: " + stopwatch.Elapsed);
            //    stopwatch.Reset();

            //    testCNN(testing_sample_size);

            //}


            testNN(testing_sample_size);
            //for (int epoch = 0; epoch < 10; epoch++) {
            //    Console.WriteLine("++++++++++++++++++++++++++++++++");
            //    Console.WriteLine("Epoch: " + epoch);
                
            //    Utils.shuffleTrainingSet();

            //    stopwatch.Start();
            //    trainNN(batchSize);
            //    testNN();
            //    stopwatch.Stop();
            //    Console.WriteLine("Time elapsed: " + stopwatch.Elapsed);
            //    stopwatch.Reset();
            //}
        }

        //static void testCNN(int partition_size) {
        //    int correct = 0;
        //    Double totalCrossEntropyLoss = 0.0;
        //    Double averageCrossEntropyLoss = 0.0;

        //    Tensor image = testing_images.partition(0, partition_size);
        //    Tensor label = testing_labels.partition(0, partition_size);

        //    for (int i = 0; i < 10000; i++) {
        //        Tuple<Double[,,], Double[,,]> t;
        //        t = CNN.forward(testImageArray[i], testLabelArray[i]);
        //        if (Utils.indexMaxValue(t.Item1) == Utils.indexMaxValue(testLabelArray[i])) {
        //            correct++;
        //        }
        //        totalCrossEntropyLoss += t.Item2[0, 0, 0];
        //    }
        //    averageCrossEntropyLoss = totalCrossEntropyLoss / 10000;

        //    Console.WriteLine(correct + " correct out of 10,000. \t Accuracy " + (Double)correct / 10000 * 100 + "%");
        //    Console.WriteLine("Average cross entropy loss: " + averageCrossEntropyLoss + "\n\n");
        //}

        //static void trainCNN(int batchSize) {
        //    for (int i = 0; i < 600; i++) {
        //        Tuple<Double[,,], Double[,,]> t;
        //        t = CNN.forward(trainImageArray [i], trainLabelArray[i]);
        //        CNN.backward();
        //        if (i % batchSize == 0 || i == 59999) {
        //            CNN.update(batchSize);
        //        }
        //    }
        //}

        static void testNN(int partition_size) {
            int correct = 0;
            Double totalCrossEntropyLoss = 0.0;
            Double averageCrossEntropyLoss = 0.0;


            int correct_tensor = 0;
            Double total_cross_entropy_loss_tensor = 0.0;

            Tuple<Tensor, Tensor> q;
            q = NN.forwardTensor(testing_images, testing_labels);
            
            for (int i=0; i < 10000; i++) {
                total_cross_entropy_loss_tensor += q.Item2.data[i];

                int index_max_value_output = -1;
                Double max_output = Double.MinValue;

                int index_max_value_label = -1;
                Double max_label = Double.MinValue;

                for (int j=0; j < q.Item1.num_channels; j++) {
                    if (q.Item1.data[i * q.Item1.num_channels + j] > max_output) {
                        max_output = q.Item1.data[i * q.Item1.num_channels + j];
                        index_max_value_output = j;
                    }
                    if (testing_labels.data[i * q.Item1.num_channels + j] > max_label) {
                        max_label = testing_labels.data[i * q.Item1.num_channels + j];
                        index_max_value_label = j;
                    }
                }
                if (index_max_value_output == index_max_value_label) {
                    correct_tensor++;
                }
            }
            Console.WriteLine(correct_tensor + " correct out of 10,000. \t Accuracy " + (Double)correct_tensor / 10000 * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + total_cross_entropy_loss_tensor / 10000);


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