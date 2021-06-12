using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using System.Threading;

namespace Conv_Net {
    static class Program {
        //[STAThread]

        public static Random rand = new Random(0);
        public static Random dropout_rand = new Random(0);
        public static Stopwatch stopwatch = new Stopwatch();

        public static Net CNN = new Net();

        public static Tensor training_images, training_labels, testing_images, testing_labels;
       
        public static int testing_sample_size = 1000;
        public static int test_batch_size = 1001;
        public static int epochs = 20;
        public static int CNN_training_sample_size = 600;
        public static int batch_size = 32;

        public static Double ALPHA = 0.01; // learning rate
        public static Double BETA_1 = 0.8; // momentum
        public static Double BETA_2 = 0.999; // RMS prop
        public static Double EPSILON = 0.00001;

        static void Main() {
            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/

            //Grad_Check.test();

            Tuple<Tensor, Tensor, Tensor, Tensor> data = IO.load_MNIST(60000, 10000, 28, 28, 1, 10);
            training_images = data.Item1;
            training_labels = data.Item2;
            testing_images = data.Item3;
            testing_labels = data.Item4;

            // CNN.load_parameters();
            test_CNN(testing_sample_size);
            for (int i = 0; i < epochs; i++) {
                Console.WriteLine("____________________________________________________________\nEPOCH: " + i);
                Utils.shuffle_training(training_images, training_labels);
                train_CNN(CNN_training_sample_size, batch_size);
                test_CNN(testing_sample_size);
                //CNN.save_parameters(i);
            }






            //Tuple<Tensor, Tensor> t;
            //t = CNN.forward(testing_images, testing_labels);
            //for (int i=0; i < 100; i++) {
            //    Utils.print_images(testing_images, i);
            //    Utils.print_labels(testing_labels, t.Item2, i);
            //}
        }


        static void test_CNN(int testing_sample_size) {
            stopwatch.Start();

            int num_batches = testing_sample_size / test_batch_size;
            int remainder = testing_sample_size - num_batches * test_batch_size;
            
            int correct = 0;
            Double total_cross_entropy_loss = 0.0;
            Tensor A, B;
            Tuple<Tensor, Tensor> R;

            

            for (int i = 0; i < num_batches; i++) {

                A = testing_images.subset(i * test_batch_size, test_batch_size);
                B = testing_labels.subset(i * test_batch_size, test_batch_size);
                
                R = CNN.forward(A, B, false);
                total_cross_entropy_loss += R.Item1.values[0];

                for (int j = 0; j < test_batch_size; j++) {
                    

                    int index_max_value_output = -1;
                    Double max_output = Double.MinValue;
                    int index_max_value_label = -1;
                    Double max_label = Double.MinValue;

                    for (int k = 0; k < R.Item2.dim_2; k++) {
                        if (R.Item2.values[j * R.Item2.dim_2 + k] > max_output) {
                            max_output = R.Item2.values[j * R.Item2.dim_2 + k];
                            index_max_value_output = j * R.Item2.dim_2 + k;
                        }
                        if (B.values[j * R.Item2.dim_2 + k] > max_label) {
                            max_label = B.values[j * R.Item2.dim_2 + k];
                            index_max_value_label = j * R.Item2.dim_2 + k;
                        }
                    }
                    if (index_max_value_output == index_max_value_label) {
                        correct++;
                    }
                }
            }
            if (remainder != 0) {
                
                A = testing_images.subset(num_batches * test_batch_size, remainder);
                B = testing_labels.subset(num_batches * test_batch_size, remainder);
               
                R = CNN.forward(A, B, false);
  
                total_cross_entropy_loss += R.Item1.values[0];

                for (int j = 0; j < remainder; j++) {
                    
                    int index_max_value_output = -1;
                    Double max_output = Double.MinValue;
                    int index_max_value_label = -1;
                    Double max_label = Double.MinValue;

                    for (int k = 0; k < R.Item2.dim_2; k++) {
                        if (R.Item2.values[j * R.Item2.dim_2 + k] > max_output) {
                            max_output = R.Item2.values[j * R.Item2.dim_2 + k];
                            index_max_value_output = j * R.Item2.dim_2 + k;
                        }
                        if (B.values[j * R.Item2.dim_2 + k] > max_label) {
                            max_label = B.values[j * R.Item2.dim_2 + k];
                            index_max_value_label = j * R.Item2.dim_2 + k;
                        }
                    }
                    if (index_max_value_output == index_max_value_label) {
                        correct++;
                    }
                }
            }

            stopwatch.Stop();
            Console.WriteLine("Testing time:\t" + stopwatch.Elapsed);
            Console.WriteLine("Accuracy:\t" + (Double)correct / testing_sample_size * 100 + "% (" + correct + " correct out of " + testing_sample_size + ")");
            Console.WriteLine("Average loss:\t" + total_cross_entropy_loss);
            stopwatch.Reset();
        }

        static void train_CNN(int training_sample_size, int batch_size) {

            stopwatch.Start();

            int num_batches = training_sample_size / batch_size;
            int remainder = training_sample_size - num_batches * batch_size;
            Tensor A;
            Tensor B;
            Tuple<Tensor, Tensor> R;

            for (int i = 0; i < num_batches; i++) {
                A = training_images.subset(i * batch_size, batch_size);
                B = training_labels.subset(i * batch_size, batch_size);
                R = CNN.forward(A, B, true);
                CNN.backward(batch_size);
                CNN.update();
            }
            if (remainder != 0) {
                A = training_images.subset(num_batches * batch_size, remainder);
                B = training_labels.subset(num_batches * batch_size, remainder);
                R = CNN.forward(A, B, true);
                CNN.backward(remainder);
                CNN.update();
            }
            stopwatch.Stop();
            Console.WriteLine("Training time:\t" + stopwatch.Elapsed + "\n");
            stopwatch.Reset();
        }
    }
}