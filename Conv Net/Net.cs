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

            // Input layer
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
            loss = Softmax.categoricalCrossEntropyLoss(target);
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

        public void update (int batchSize) {
            FC3.update(batchSize);
            FC2.update(batchSize);
            FC1.update(batchSize);
        }
    }
}
