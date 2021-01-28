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
            Input = new InputLayer(28, 28, 1);
            Flatten = new FlattenLayer();
            FC1 = new FullyConnectedLayer(784, 5);
            Relu1 = new ReluLayer();
            FC2 = new FullyConnectedLayer(5, 6);
            Relu2 = new ReluLayer();
            FC3 = new FullyConnectedLayer(6, 10);
            Softmax = new SoftmaxLayer();
            Loss = new LossLayer();
        }

        public Double[,,] forward (Double[,,] input) {
            Double[,,] output = Input.forward(input);
            output = Flatten.forward(output);

            output = FC1.forward(output);
            output = Relu1.forward(output);

            output = FC2.forward(output);
            output = Relu2.forward(output);

            output = FC3.forward(output);
            output = Softmax.forward(output);

            return output;
        }
        
        public Double[,,] loss (Double [,,] input, Double[,,] target) {
            return Loss.forward(input, target);
        }

        public void backward () {
            Double[,,] gradient;
            
            // for output layer
            gradient = Loss.backward();
            gradient = Softmax.backward(gradient);
            gradient = FC3.backward(gradient);

            // Hidden layer 2
            gradient = Relu2.backward(gradient);
            gradient = FC2.backward(gradient);

            // Hidden layer 1
            gradient = Relu1.backward(gradient);
            FC1.storeGradient(gradient);
        }

        public void update () {
            FC3.update();
            FC2.update();
            FC1.update();
        }
    }
}
