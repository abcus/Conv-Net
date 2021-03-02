using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class ConvNet {

        public Input_Layer Input;
        public Convolution_Layer Conv1, Conv2;
        public Relu_Layer Relu1, Relu2;
        public Max_Pooling_Layer Pool1, Pool2;
        public Flatten_Layer Flatten3;
        public Fully_Connected_Layer FC3;
        public Softmax_Loss_Layer Softmax;

        public ConvNet () {
            
            // Input layer
            Input = new Input_Layer(28, 28, 1);

            // Conv layer 1
            Conv1 = new Convolution_Layer(1, 8, 5, 5, false); // 24 x 24 x 8 (200 F + 8 B)
            Relu1 = new Relu_Layer();
            Pool1 = new Max_Pooling_Layer(2, 2, 2); // 12 x 12 x 8

            // Conv layer 2
            Conv2 = new Convolution_Layer(8, 8, 5, 5, true); // 8 x 8 x 8 (1600 F + 8 B)
            Relu2 = new Relu_Layer();
            Pool2 = new Max_Pooling_Layer(2, 2, 2);  // 4 x 4 x 8

            // Fully connected layer 2
            Flatten3 = new Flatten_Layer(); // 1 x 1 x 128
            FC3 = new Fully_Connected_Layer(4 * 4 * 8, 10, true); // 1 x 1 x 10 (1280 W + 1 B)
            Softmax = new Softmax_Loss_Layer();
        }

        public Tuple<Tensor, Tensor> forward (Double[,,] input, Double[,,] target) {
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

            return Tuple.Create(Utils.label_to_tensor(loss), Utils.label_to_tensor(output));
        }

        public Tuple<Tensor, Tensor> forward_tensor (Tensor input, Tensor target) {
            Tensor output;
            Tensor loss;

            output = Conv1.forward_tensor(input);
            output = Relu1.forward_tensor(output);
            output = Pool1.forward_tensor(output);

            output = Conv2.forward_tensor(output);
            output = Relu2.forward_tensor(output);
            output = Pool2.forward_tensor(output);

            output = Flatten3.forward_tensor(output);
            output = FC3.forward_tensor(output);
            output = Softmax.forward_tensor(output);

            loss = Softmax.loss_tensor(target);

            return Tuple.Create(loss, output);
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
