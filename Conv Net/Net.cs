using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Net {

        public InputLayer Input;
        public FlattenLayer Flatten;
        public FullyConnectedLayer FC1, FC2, FC3;
        public ReluLayer Relu1, Relu2;
        public SoftmaxLossLayer Softmax;

        public Net () {

            // Input layer
            Input = new InputLayer(28, 28, 1);
            Flatten = new FlattenLayer();
            
            // Hidden layer 1
            FC1 = new FullyConnectedLayer(784, 5, false);
            Relu1 = new ReluLayer();
            
            // Hidden layer 2
            FC2 = new FullyConnectedLayer(5, 6, true);
            Relu2 = new ReluLayer();
            
            // Hidden layer 3
            FC3 = new FullyConnectedLayer(6, 10, true);
            Softmax = new SoftmaxLossLayer();
        }

        public Tuple<Double[,,], Double[,,]> forward (Double[,,] input, Double[,,] target) {

            Double[,,] output;
            Double[,,] loss;
            
            // Input and flatten layer
            output = Input.forward(input);
            output = Flatten.forward(output);

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

        public Tuple<Tensor, Tensor> forward_tensor(Tensor input, Tensor target) {
            Tensor output;
            Tensor loss;

            // Input and flatten layer
            output = Flatten.forward_tensor(input);

            // Hidden layer 1
            output = FC1.forward_tensor(output);
            output = Relu1.forward_tensor(output);

            // Hidden layer 2
            output = FC2.forward_tensor(output);
            output = Relu2.forward_tensor(output);

            // Output layer
            output = FC3.forward_tensor(output);
            output = Softmax.forward_tensor(output);

            // Loss layer
            loss = Softmax.loss_tensor(target);
            return Tuple.Create(output, loss);
        }


        public void backward () {
            
            Double[,,] grad;

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

        public void backward_tensor () {
            Tensor grad;

            grad = Softmax.backward_tensor();
            grad = FC3.backward_tensor(grad);
            // Console.WriteLine(FC3.gradient_biases_tensor);
            // Console.WriteLine(FC3.gradient_weights_tensor);

            grad = Relu2.backward_tensor(grad);
            grad = FC2.backward_tensor(grad);
            //Console.WriteLine(FC2.gradient_biases_tensor);
            // Console.WriteLine(FC2.gradient_weights_tensor);

            grad = Relu1.backward_tensor(grad);
            grad = FC1.backward_tensor(grad);
            //Console.WriteLine(FC1.gradient_biases_tensor);
            //Console.WriteLine(FC1.gradient_weights_tensor);
        }

        public void update (int batchSize) {
            FC3.update(batchSize);
            FC2.update(batchSize);
            FC1.update(batchSize);
        }

        public void update_tensor (int batchSize) {
            FC3.update_tensor(batchSize);
            FC2.update_tensor(batchSize);
            FC1.update_tensor(batchSize);
        }
    }
}
