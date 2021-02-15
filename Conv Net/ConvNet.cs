using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class ConvNet {

        public InputLayer Input;
        public ConvolutionLayer Conv1, Conv2;
        public ReluLayer Relu1, Relu2;
        public MaxPoolingLayer Pool1, Pool2;
        public FlattenLayer Flatten3;
        public FullyConnectedLayer FC3;
        public SoftmaxLossLayer Softmax;

        public ConvNet () {
            
            // Input layer
            Input = new InputLayer(28, 28, 1);

            // Conv layer 1
            Conv1 = new ConvolutionLayer(1, 8, 5, 5, false); // 24 x 24 x 8 (200 F + 8 B)
            Relu1 = new ReluLayer();
            Pool1 = new MaxPoolingLayer(2, 2, 2); // 12 x 12 x 8

            // Conv layer 2
            Conv2 = new ConvolutionLayer(8, 8, 5, 5, true); // 8 x 8 x 8 (1600 F + 8 B)
            Relu2 = new ReluLayer();
            Pool2 = new MaxPoolingLayer(2, 2, 2);  // 4 x 4 x 8

            // Fully connected layer 2
            Flatten3 = new FlattenLayer(); // 1 x 1 x 128
            FC3 = new FullyConnectedLayer(4 * 4 * 8, 10, true); // 1 x 1 x 10 (1280 W + 1 B)
            Softmax = new SoftmaxLossLayer();
        }

        public Tuple<Double[,,], Double[,,]> forward (Double[,,] input, Double[,,] target) {

            Double[,,] output;
            Double[,,] loss;

            output = Input.forward(input);

            output = Conv1.forward(output);
            output = Relu1.forward(output);
            output = Pool1.forward(output);

            output = Conv2.forward(output);
            output = Relu2.forward(output);
            output = Pool2.forward(output);
            
            output = Flatten3.forward(output);
            output = FC3.forward(output);
            output = Softmax.forward(output);

            loss = Softmax.loss(target);
            return Tuple.Create(output, loss);
        }

        public void backward () {
            Double[,,] grad;

            grad = Softmax.backward();
            grad = FC3.backward(grad);
            grad = Flatten3.backward(grad);

            grad = Pool2.backward(grad);
            grad = Relu2.backward(grad);
            grad = Conv2.backward(grad);

            grad = Pool1.backward(grad);
            grad = Relu1.backward(grad);
            grad = Conv1.backward(grad);
        }
         
        public void update (int batchSize) {
            FC3.update(batchSize);
            Conv2.update(batchSize);
            Conv1.update(batchSize);
        }
            
            

        
        

    }
}
