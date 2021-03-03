﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Net {

        public Input_Layer Input;
        public Flatten_Layer Flatten;
        public Fully_Connected_Layer FC1, FC2, FC3;
        public Relu_Layer Relu1, Relu2;
        public Softmax_Loss_Layer Softmax;

        public Net () {

            // Input layer
            Input = new Input_Layer(28, 28, 1);
            Flatten = new Flatten_Layer();
            
            // Hidden layer 1
            FC1 = new Fully_Connected_Layer(784, 5, false);
            Relu1 = new Relu_Layer();
            
            // Hidden layer 2
            FC2 = new Fully_Connected_Layer(5, 6, true);
            Relu2 = new Relu_Layer();
            
            // Hidden layer 3
            FC3 = new Fully_Connected_Layer(6, 10, true);
            Softmax = new Softmax_Loss_Layer();
        }

        public Tuple<Tensor, Tensor> forward(Tensor input, Tensor target) {
            Tensor output;
            Tensor loss;

            // Input and flatten layer
            output = Flatten.forward(input);

            // Hidden layer 1
            output = FC1.forward(output);
            output = Relu1.forward(output);

            // Hidden layer 2
            output = FC2.forward(output);
            output = Relu2.forward(output);

            // Output layer
            output = FC3.forward(output);
            output = Softmax.forward(output);

            // Loss layer
            loss = Softmax.loss(target);
            return Tuple.Create(output, loss);
        }

        public void backward () {
            Tensor grad;

            // Output layer
            grad = Softmax.backward();
            grad = FC3.backward(grad);

            // Hidden layer 2
            grad = Relu2.backward(grad);
            grad = FC2.backward(grad);

            // Hidden layer 1 
            // FC1.backward returns null as gradient of loss with respect to FC1 inputs (which is the image) is not needed
            grad = Relu1.backward(grad);
            grad = FC1.backward(grad);
        }

        public void update (int batch_size) {
            FC3.update_tensor(batch_size);
            FC2.update_tensor(batch_size);
            FC1.update_tensor(batch_size);
        }
    }
}
