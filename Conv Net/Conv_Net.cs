using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Conv_Net {
    class Conv_Net {

        public Input_Layer Input;
        public Convolution_Layer Conv_1, Conv_2;
        public Relu_Layer Relu_1, Relu_2;
        public Max_Pooling_Layer Pool_1, Pool_2;
        public Dropout_Layer Dropout_1, Dropout_2;
        public Flatten_Layer Flatten_3;
        public Fully_Connected_Layer FC_3;
        public Softmax_Loss_Layer Softmax;

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
        public Conv_Net () {

            Input = new Input_Layer();

            Conv_1 = new Convolution_Layer(1, 32, 3, 3, false, 1, 1, 1); 
            Relu_1 = new Relu_Layer();
            Pool_1 = new Max_Pooling_Layer(2, 2, 2);
            Dropout_1 = new Dropout_Layer(0.2);

            Conv_2 = new Convolution_Layer(32, 64, 4, 4, true, 1, 2, 3); 
            Relu_2 = new Relu_Layer();
            Pool_2 = new Max_Pooling_Layer(2, 2, 2);
            Dropout_2 = new Dropout_Layer(0.2);

            Flatten_3 = new Flatten_Layer(); 
            FC_3 = new Fully_Connected_Layer(2 * 2 * 64, 10, true); 
            Softmax = new Softmax_Loss_Layer();

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
            Tensor output;
            Tensor loss;

            output = Input.forward(input);

            output = Conv_1.forward(output);
            output = Relu_1.forward(output);
            output = Pool_1.forward(output);
            if (is_train == true) { output = Dropout_1.forward(output); }

            output = Conv_2.forward(output);
            output = Relu_2.forward(output);
            output = Pool_2.forward(output);
            if (is_train == true) { output = Dropout_2.forward(output); }

            output = Flatten_3.forward(output);
            output = FC_3.forward(output);
            output = Softmax.forward(output);

            loss = Softmax.loss(target);

            return Tuple.Create(loss, output);
        }

        /// <summary>
        /// Backpropagation for CNN
        /// Output of Conv_1 is null (dL/dI is not needed because Conv_1 is the first layer)
        /// </summary>
        public void backward(int batch_size) {
            Tensor grad = new Tensor (0, 0, 0, 0, 0);

            grad = Softmax.backward();
            grad = FC_3.backward(grad);
            grad = Flatten_3.backward(grad);

            grad = Dropout_2.backward(grad);
            grad = Pool_2.backward(grad);
            grad = Relu_2.backward(grad);
            grad = Conv_2.backward(grad);

            grad = Dropout_1.backward(grad);
            grad = Pool_1.backward(grad);
            grad = Relu_1.backward(grad);
            grad = Conv_1.backward(grad);
        }

        public void update () {
            Optim.SGD_FC(FC_3);
            Optim.SGD_Conv(Conv_2);
            Optim.SGD_Conv(Conv_1);
            Optim.t += 1; // iterate t for bias correction
        }

        public void save_parameters(int epoch) {
            StreamWriter writer = new StreamWriter("parameters " + epoch + ".txt", false);

            foreach(Double b in Conv_1.B.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in Conv_1.F.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in Conv_2.B.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in Conv_2.F.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in FC_3.B.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in FC_3.W.values) {
                writer.WriteLine(b);
            }
            writer.Close();
        }


        public void load_parameters() {
            System.IO.StreamReader reader = new System.IO.StreamReader(@"parameters 997.txt");

            for (int i=0; i < Conv_1.B.values.Length; i++) {
                Conv_1.B.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < Conv_1.F.values.Length; i++) {
                Conv_1.F.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < Conv_2.B.values.Length; i++) {
                Conv_2.B.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < Conv_2.F.values.Length; i++) {
                Conv_2.F.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < FC_3.B.values.Length; i++) {
                FC_3.B.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < FC_3.W.values.Length; i++) {
                FC_3.W.values[i] = Convert.ToDouble(reader.ReadLine());
            }
        }
    }
}
