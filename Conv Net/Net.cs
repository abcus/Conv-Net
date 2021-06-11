using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Conv_Net {
    class Net {

        public Base_Layer[] layer_list = new Base_Layer[14];
        public Optimizer Optim;

        /// <summary>
        /// Conv layer 1
        /// Input of Conv_1: [batch size x 28 x 28 x 1]
        /// Output of Conv_1: [batch size x 24 x 24 x 16]
        /// Output of Pool_1: [batch size x 12 x 12 x 16]
        ///
        /// Conv layer 2
        /// Input of Conv_2: [batch size x 12 x 12 x 16]
        /// Output of Conv_2: [batch size x 8 x 8 x 32]
        /// Output of Pool_2: [batch size x 4 x 4 x 32]
        ///
        /// Fully connected layer 3
        /// Input of Flatten_3: [batch size x 4 x 4 x 32]
        /// Output of Flatten 3: [batch size x 512 x 1 x 1]
        /// Output of FC_3: [batch size x 10 x 1 x 1]
        /// Output of Softmax: [batch size x 10 x 1 x 1]
        /// </summary>
        public Net () {

            layer_list[0] = new Input_Layer();
            layer_list[1] = new Convolution_Layer(1, 16, 5, 5, false, 0, 1, 1);
            layer_list[2] = new Batch_Normalization_Layer(16);
            layer_list[3] = new Relu_Layer();
            layer_list[4] = new Max_Pooling_Layer(2, 2, 2);
            layer_list[5] = new Dropout_Layer(0.2);

            layer_list[6] = new Convolution_Layer(16, 32, 5, 5, true, 0, 1, 1);
            layer_list[7] = new Batch_Normalization_Layer(32);
            layer_list[8] = new Relu_Layer();
            layer_list[9] = new Max_Pooling_Layer(2, 2, 2);
            layer_list[10] = new Dropout_Layer(0.2);

            layer_list[11] = new Flatten_Layer(); 
            layer_list[12] = new Fully_Connected_Layer(4 * 4 * 32, 10, true); 
            layer_list[13] = new Softmax_Loss_Layer();

            Optim = new Optimizer();
        }

        /// <summary>
        /// If training, then apply dropout
        /// </summary>
        /// <param name="input"></param>
        /// <param name="target"></param>
        /// <param name="is_train"></param>
        /// <returns></returns>
        public Tuple<Tensor, Tensor> forward (Tensor input, Tensor target, bool is_train) {
            Tensor output = input;
            Tensor loss;

            for (int i=0; i < layer_list.Length; i++) {
                if (layer_list[i].test_train_mode == false) {
                    output = layer_list[i].forward(output);
                } else {
                    output = layer_list[i].forward(output, is_train);
                }
            }
            loss = layer_list[layer_list.Length - 1].loss(target);
            return Tuple.Create(loss, output);
        }

        /// <summary>
        /// Backpropagation for CNN
        /// Output of Conv_1 is null (dL/dI is not needed because Conv_1 is the first layer)
        /// </summary>
        public void backward(int batch_size) {
            Tensor grad = new Tensor(1, 1);
            for (int i = layer_list.Length - 1; i >= 0; i--) {
                grad = layer_list[i].backward(grad);
            }
        }

        public void update () {
            for (int i=0; i < layer_list.Length; i++) {
                if (layer_list[i].trainable_parameters == true) {
                    Optim.Momentum(layer_list[i]);
                }
            }
            Optim.t += 1; // iterate t for bias correction
        }

        //public void save_parameters(int epoch) {
        //    StreamWriter writer = new StreamWriter("parameters " + epoch + ".txt", false);

        //    foreach(Double b in Conv_1.B.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in Conv_1.W.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach(Double b in BN_1.B.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in BN_1.W.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in Conv_2.B.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in Conv_2.W.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in BN_2.B.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in BN_2.W.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in FC_3.B.values) {
        //        writer.WriteLine(b);
        //    }
        //    foreach (Double b in FC_3.W.values) {
        //        writer.WriteLine(b);
        //    }
        //    writer.Close();
        //}


        //public void load_parameters() {
        //    System.IO.StreamReader reader = new System.IO.StreamReader(@"parameters 997.txt");

        //    for (int i=0; i < Conv_1.B.values.Length; i++) {
        //        Conv_1.B.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < Conv_1.W.values.Length; i++) {
        //        Conv_1.W.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i=0; i < BN_1.B.values.Length; i++) {
        //        BN_1.B.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < BN_1.W.values.Length; i++) {
        //        BN_1.W.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < Conv_2.B.values.Length; i++) {
        //        Conv_2.B.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < Conv_2.W.values.Length; i++) {
        //        Conv_2.W.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < BN_2.B.values.Length; i++) {
        //        BN_2.B.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < BN_2.W.values.Length; i++) {
        //        BN_2.W.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < FC_3.B.values.Length; i++) {
        //        FC_3.B.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //    for (int i = 0; i < FC_3.W.values.Length; i++) {
        //        FC_3.W.values[i] = Convert.ToDouble(reader.ReadLine());
        //    }
        //}
    }
}
