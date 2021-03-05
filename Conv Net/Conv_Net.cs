using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Conv_Net {
    class Conv_Net {

        public Pad Pad_1, Pad_2;

        public Input_Layer Input;
        public Convolution_Layer Conv_1, Conv_2;
        public Relu_Layer Relu_1, Relu_2;
        public Max_Pooling_Layer Pool_1, Pool_2;
        public Flatten_Layer Flatten_3;
        public Fully_Connected_Layer FC_3;
        public Softmax_Loss_Layer Softmax;

        /// <summary>
        /// Conv layer 1
        /// Input of Conv_1: [batch size x 28 x 28 x 1]
        /// Output of Conv_1: [batch size x 24 x 24 x 8]
        /// Output of Pool_1: [batch size x 12 x 12 x 8]
        ///
        /// Conv layer 2
        /// Input of Conv_2: [batch size x 12 x 12 x 8]
        /// Output of Conv_2: [batch size x 8 x 8 x 8]
        /// Output of Pool_2: [batch size x 4 x 4 x 8]
        ///
        /// Fully connected layer 3
        /// Input of Flatten_3: [batch size x 4 x 4 x 8]
        /// Output of Flatten 3: [batch size x 128 x 1 x 1]
        /// Output of FC_3: [batch size x 10 x 1 x 1]
        /// Output of Softmax: [batch size x 10 x 1 x 1]
        /// </summary>
        public Conv_Net () {

            Input = new Input_Layer();

            Pad_1 = new Pad(2);
            Conv_1 = new Convolution_Layer(1, 8, 5, 5, 2, false); 
            Relu_1 = new Relu_Layer();
            Pool_1 = new Max_Pooling_Layer(2, 2, 2);

            Pad_2 = new Pad(2);
            Conv_2 = new Convolution_Layer(8, 8, 5, 5, 2, true); 
            Relu_2 = new Relu_Layer();
            Pool_2 = new Max_Pooling_Layer(2, 2, 2);  
            
            Flatten_3 = new Flatten_Layer(); 
            FC_3 = new Fully_Connected_Layer(7 * 7 * 8, 10, true); 
            Softmax = new Softmax_Loss_Layer();
        }

        public Tuple<Tensor, Tensor> forward (Tensor input, Tensor target) {
            Tensor output;
            Tensor loss;

            output = Input.forward(input);

            output = Pad_1.forward(output);
            output = Conv_1.forward(output);
            output = Relu_1.forward(output);
            output = Pool_1.forward(output);

            output = Pad_2.forward(output);
            output = Conv_2.forward(output);
            output = Relu_2.forward(output);
            output = Pool_2.forward(output);

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
            Tensor grad;

            grad = Softmax.backward(batch_size);
            grad = FC_3.backward(grad);
            grad = Flatten_3.backward(grad);

            grad = Pool_2.backward(grad);
            grad = Relu_2.backward(grad);
            grad = Conv_2.backward(grad);
            grad = Pad_2.backward(grad);

            grad = Pool_1.backward(grad);
            grad = Relu_1.backward(grad);
            grad = Conv_1.backward(grad);
        }

        public void update () {
            FC_3.update();
            Conv_2.update();
            Conv_1.update();
        }

        public void save_parameters(int epoch) {
            StreamWriter writer = new StreamWriter("parameters " + epoch + ".txt", false);

            foreach(Double b in Conv_1.biases.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in Conv_1.filters.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in Conv_2.biases.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in Conv_2.filters.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in FC_3.biases.values) {
                writer.WriteLine(b);
            }
            foreach (Double b in FC_3.weights.values) {
                writer.WriteLine(b);
            }
            writer.Close();
        }


        public void load_parameters() {
            System.IO.StreamReader reader = new System.IO.StreamReader(@"parameters 15.txt");

            for (int i=0; i < Conv_1.biases.values.Length; i++) {
                Conv_1.biases.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < Conv_1.filters.values.Length; i++) {
                Conv_1.filters.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < Conv_2.biases.values.Length; i++) {
                Conv_2.biases.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < Conv_2.filters.values.Length; i++) {
                Conv_2.filters.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < FC_3.biases.values.Length; i++) {
                FC_3.biases.values[i] = Convert.ToDouble(reader.ReadLine());
            }
            for (int i = 0; i < FC_3.weights.values.Length; i++) {
                FC_3.weights.values[i] = Convert.ToDouble(reader.ReadLine());
            }
        }
    }
}
