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
        public SoftmaxLayer Softmax;
        public LossLayer Loss;

        public Net () {

            // Input layer
            Input = new InputLayer(28, 28, 1);
            Flatten = new FlattenLayer();
            
            // Hidden layer 1
            FC1 = new FullyConnectedLayer(784, 5);
            Relu1 = new ReluLayer();
            
            // Hidden layer 2
            FC2 = new FullyConnectedLayer(5, 6);
            Relu2 = new ReluLayer();
            
            // Hidden layer 3
            FC3 = new FullyConnectedLayer(6, 10);
            Softmax = new SoftmaxLayer();
            Loss = new LossLayer();
        }

        public Double[,,] forward (Double[,,] input) {
            
            // Input layer
            Double[,,] output = Input.forward(input);
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

            return output;
        }
        
        public Double[,,] loss (Double [,,] input, Double[,,] target) {
            return Loss.forward(input, target);
        }

        public void backward () {
            Double[,,] grad;
            
            // Output layer
            grad = Loss.backward();
            grad = Softmax.backward(grad);
            grad = FC3.backward(grad);

            // Hidden layer 2
            grad = Relu2.backward(grad);
            grad = FC2.backward(grad);

            // Hidden layer 1
            grad = Relu1.backward(grad);
            FC1.storeGradient(grad);
        }

        public void update () {
            FC3.update();
            FC2.update();
            FC1.update();
        }
    }
}
