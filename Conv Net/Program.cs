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

        public static Tensor training_images;
        public static Tensor training_labels;
        public static Tensor testing_images;
        public static Tensor testing_labels;

        public static Random rand = new Random(0);
        public static MathNet.Numerics.Distributions.Normal normalDist = new MathNet.Numerics.Distributions.Normal(0, 1, rand);
        public static Stopwatch stopwatch = new Stopwatch();

        // public static Net NN = new Net();
        public static Conv_Net CNN = new Conv_Net();
        public static Double eta = 0.01;
        public static int batchSize = 16;

        public static int testing_sample_size = 10000;
        public static int training_sample_size = 60000;
        public static int CNN_training_sample_size = 600;

        static void Main() {

            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            Tuple<Tensor, Tensor, Tensor, Tensor> data = Utils.load_MNIST(60000, 10000, 28, 28, 1, 10);
            training_images = data.Item1;
            training_labels = data.Item2;
            testing_images = data.Item3;
            testing_labels = data.Item4;

            test_CNN(testing_sample_size);
            for (int epoch = 0; epoch < 20; epoch++) {
                Console.WriteLine("------------------------------------------");
                Console.WriteLine("Epoch: " + epoch);

                Utils.shuffle_Tensor(training_images, training_labels);

                stopwatch.Start();
                train_CNN(CNN_training_sample_size, batchSize);
                stopwatch.Stop();
                Console.WriteLine("Time elapsed for training: " + stopwatch.Elapsed);
                stopwatch.Reset();

                test_CNN(testing_sample_size);
            }

            //test_NN(testing_sample_size);
            //for (int epoch = 0; epoch < 10; epoch++) {
            //    Console.WriteLine("++++++++++++++++++++++++++++++++");
            //    Console.WriteLine("Epoch: " + epoch);

            //    Utils.shuffle_Tensor(training_images, training_labels);

            //    stopwatch.Start();
            //    train_NN(training_sample_size, batchSize);
            //    test_NN(testing_sample_size);
            //    stopwatch.Stop();
            //    Console.WriteLine("Time elapsed: " + stopwatch.Elapsed);
            //    stopwatch.Reset();
            //}
        }


        static void test_CNN(int testing_sample_size) {
            int correct = 0;
            Double total_cross_entropy_loss = 0.0;
            Tuple<Tensor, Tensor> t;

            t = CNN.forward(testing_images, testing_labels);

            for (int i = 0; i < testing_sample_size; i++) {
                total_cross_entropy_loss += t.Item1.values[i];

                int index_max_value_output = -1;
                Double max_output = Double.MinValue;

                int index_max_value_label = -1;
                Double max_label = Double.MinValue;

                for (int j = 0; j < t.Item2.dim_2; j++) {
                    if (t.Item2.values[i * t.Item2.dim_2 + j] > max_output) {
                        max_output = t.Item2.values[i * t.Item2.dim_2 + j];
                        index_max_value_output = j;
                    }
                    if (testing_labels.values[i * t.Item2.dim_2 + j] > index_max_value_label) {
                        max_label = testing_labels.values[i * t.Item2.dim_2 + j];
                        index_max_value_label = j;
                    }
                }
                if (index_max_value_output == index_max_value_label) {
                    correct++;
                }
            }
            Console.WriteLine(correct + " correct out of " + testing_sample_size + ". \t Accuracy " + (Double)correct / testing_sample_size * 100 + "%");
            Console.WriteLine("Average cross entropy loss: " + total_cross_entropy_loss / testing_sample_size);
        }

        static void train_CNN(int training_sample_size, int batch_size) {
            int num_batches = training_sample_size / batch_size;
            int remainder = training_sample_size - num_batches * batch_size;
            Tensor A;
            Tensor B;
            Tuple<Tensor, Tensor> R;

            for (int i = 0; i < num_batches; i++) {
                A = training_images.subset(i * batch_size, batch_size);
                B = training_labels.subset(i * batch_size, batch_size);
                R = CNN.forward(A, B);
                CNN.backward();
                CNN.update(batch_size);
            }
            if (remainder != 0) {
                A = training_images.subset(num_batches * batch_size, remainder);
                B = training_labels.subset(num_batches * batch_size, remainder);
                R = CNN.forward(A, B);
                CNN.backward();
                CNN.update(remainder);
            }
        }

        //static void test_NN(int testing_sample_size) {

        //    int correct = 0;
        //    Double total_cross_entropy_loss = 0.0;
        //    Tuple<Tensor, Tensor> t;

        //    t = NN.forward(testing_images, testing_labels);

        //    for (int i = 0; i < testing_sample_size; i++) {
        //        total_cross_entropy_loss += t.Item2.values[i];

        //        int index_max_value_output = -1;
        //        Double max_output = Double.MinValue;

        //        int index_max_value_label = -1;
        //        Double max_label = Double.MinValue;

        //        for (int j = 0; j < t.Item1.dim_2; j++) {
        //            if (t.Item1.values[i * t.Item1.dim_2 + j] > max_output) {
        //                max_output = t.Item1.values[i * t.Item1.dim_2 + j];
        //                index_max_value_output = j;
        //            }
        //            if (testing_labels.values[i * t.Item1.dim_2 + j] > max_label) {
        //                max_label = testing_labels.values[i * t.Item1.dim_2 + j];
        //                index_max_value_label = j;
        //            }
        //        }
        //        if (index_max_value_output == index_max_value_label) {
        //            correct++;
        //        }
        //    }
        //    Console.WriteLine(correct + " correct out of " + testing_sample_size + ". \t Accuracy " + (Double)correct / testing_sample_size * 100 + "%");
        //    Console.WriteLine("Average cross entropy loss: " + total_cross_entropy_loss / testing_sample_size);
        //}

        //static void train_NN(int training_sample_size, int batch_size) {
        //    int num_batches = training_sample_size / batch_size;
        //    int remainder = training_sample_size - num_batches * batch_size;
        //    Tensor A;
        //    Tensor B;
        //    Tuple<Tensor, Tensor> R;

        //    for (int i = 0; i < num_batches; i++) {
        //        A = training_images.subset(i * batch_size, batch_size);
        //        B = training_labels.subset(i * batch_size, batch_size);
        //        R = NN.forward(A, B);
        //        NN.backward();
        //        NN.update(batch_size);
        //    }
        //    if (remainder != 0) {
        //        A = training_images.subset(num_batches * batch_size, remainder);
        //        B = training_labels.subset(num_batches * batch_size, remainder);
        //        R = NN.forward(A, B);
        //        NN.backward();
        //        NN.update(remainder);
        //    }
        //}
    }
}