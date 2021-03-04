using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Net {

        public Input_Layer Input;
        public Flatten_Layer Flatten;
        public Fully_Connected_Layer FC_1, FC_2, FC_3;
        public Relu_Layer Relu_1, Relu_2;
        public Softmax_Loss_Layer Softmax;

        public Net () {

            // Input layer
            Input = new Input_Layer();
            Flatten = new Flatten_Layer();
            
            // Hidden layer 1
            FC_1 = new Fully_Connected_Layer(784, 5, false);
            Relu_1 = new Relu_Layer();
            
            // Hidden layer 2
            FC_2 = new Fully_Connected_Layer(5, 6, true);
            Relu_2 = new Relu_Layer();
            
            // Hidden layer 3
            FC_3 = new Fully_Connected_Layer(6, 10, true);
            Softmax = new Softmax_Loss_Layer();
        }

        public Tuple<Tensor, Tensor> forward(Tensor input, Tensor target) {
            Tensor output;
            Tensor loss;

            // Input and flatten layer
            output = Input.forward(input);
            output = Flatten.forward(output);

            // Hidden layer 1
            output = FC_1.forward(output);
            output = Relu_1.forward(output);

            // Hidden layer 2
            output = FC_2.forward(output);
            output = Relu_2.forward(output);

            // Output layer
            output = FC_3.forward(output);
            output = Softmax.forward(output);

            // Loss layer
            loss = Softmax.loss(target);
            return Tuple.Create(output, loss);
        }

        public void backward (int batch_size) {
            Tensor grad;

            // Output layer
            grad = Softmax.backward(batch_size);
            grad = FC_3.backward(grad);

            // Hidden layer 2
            grad = Relu_2.backward(grad);
            grad = FC_2.backward(grad);

            // Hidden layer 1 
            // FC1.backward returns null as gradient of loss with respect to FC1 inputs (which is the image) is not needed
            grad = Relu_1.backward(grad);
            grad = FC_1.backward(grad);
        }

        public void update () {
            FC_3.update();
            FC_2.update();
            FC_1.update();
        }
    }
}
