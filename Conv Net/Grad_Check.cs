using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Conv_Net {
    
    static class Grad_Check {

        /// <summary>
        /// Returns analytic gradient
        /// </summary>
        /// <param name="layer_1"></param>
        /// <param name="layer_2"> set as Input_Layer to test layer_1 only </param>
        /// <param name="loss_layer"></param>
        /// <param name="I"></param>
        /// <param name="T"></param>
        /// <returns > Tuple of analytic gradients <dB, dW, dI> </returns>
        public static Tuple<Tensor, Tensor, Tensor> analytic_grad(Layer layer_1, Layer layer_2, Layer loss_layer, Tensor I, Tensor T) {
            Tensor I_copy = Utils.copy(I);
            Tensor analytic_dI, analytic_dB, analytic_dW;

            loss_layer.loss(layer_2.forward(layer_1.forward(I_copy), true), T);
            analytic_dI = layer_1.backward(layer_2.backward(loss_layer.backward()));
            analytic_dB = layer_1.dB;
            analytic_dW = layer_1.dW;

            return Tuple.Create(analytic_dB, analytic_dW, analytic_dI);
        }
        /// <summary>
        /// Returns numerical gradient
        /// </summary>
        /// <param name="test_layer_1"></param>
        /// <param name="test_layer_2"> set as Input_Layer to test layer_1 only </param>
        /// <param name="loss_layer"></param>
        /// <param name="I"></param>
        /// <param name="T"></param>
        /// <param name="h"></param>
        /// <returns> Tuple of analytic gradients <dB, dW, dI> </returns>
        public static Tuple<Tensor, Tensor, Tensor> numeric_grad(Layer test_layer_1, Layer test_layer_2, Layer loss_layer, Tensor I, Tensor T, Double h = 0.00001) {
            Tensor I_copy; 
            Tensor B = test_layer_1.B;
            Tensor W = test_layer_1.W;
            Tensor numeric_dB = new Tensor(B.dimensions, B.dim_1, B.dim_2, B.dim_3, B.dim_4);
            Tensor numeric_dW = new Tensor(W.dimensions, W.dim_1, W.dim_2, W.dim_3, W.dim_4);
            Tensor numeric_dI = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);

            for (int i=0; i < B.values.Length; i++) {
                I_copy = Utils.copy(I);
                test_layer_1.B.values[i] += h;
                Tensor L_up = loss_layer.loss(test_layer_2.forward(test_layer_1.forward(I_copy), true), T);

                I_copy = Utils.copy(I);
                test_layer_1.B.values[i] -= 2 * h;
                Tensor L_down = loss_layer.loss(test_layer_2.forward(test_layer_1.forward(I_copy), true), T);

                test_layer_1.B.values[i] += h;

                numeric_dB.values[i] = Utils.sum(Utils.subtract(L_up, L_down)) / (2 * h);
            }

            for (int i=0; i < W.values.Length; i++) {
                I_copy = Utils.copy(I);
                test_layer_1.W.values[i] += h;
                Tensor L_up = loss_layer.loss(test_layer_2.forward(test_layer_1.forward(I_copy), true), T);

                I_copy = Utils.copy(I);
                test_layer_1.W.values[i] -= 2 * h;
                Tensor L_down = loss_layer.loss(test_layer_2.forward(test_layer_1.forward(I_copy), true), T);

                test_layer_1.W.values[i] += h;

                numeric_dW.values[i] = Utils.sum(Utils.subtract(L_up, L_down)) / (2 * h);
            }

            for (int i = 0; i < I.values.Length; i++) {
                Tensor I_up = Utils.copy(I);
                Tensor I_down = Utils.copy(I);

                I_up.values[i] += h;
                I_down.values[i] -= h;

                Tensor L_up = loss_layer.loss(test_layer_2.forward(test_layer_1.forward(I_up), true), T);
                Tensor L_down = loss_layer.loss(test_layer_2.forward(test_layer_1.forward(I_down), true), T);

                numeric_dI.values[i] = Utils.sum(Utils.subtract(L_up, L_down)) / (2 * h);
            }
            return Tuple.Create(numeric_dB, numeric_dW, numeric_dI);
        }

        
        public static void test () {

            Random rand = new Random(0);

            // Batch norm test
            int n = 8; int d = 3;

            Tensor I_BN = new Tensor(2, n, d);
            Tensor T_BN = new Tensor(2, n, d);
            for (int i = 0; i < I_BN.values.Length; i++) { I_BN.values[i] = rand.NextDouble(); }
            for (int i = 0; i < T_BN.values.Length; i++) { T_BN.values[i] = rand.NextDouble(); }

            // Conv test
            int I_samples = 2; int I_rows = 8; int I_columns = 8; int I_channels = 4;
            int F_num = 3; int F_rows = 3; int F_columns = 3; int F_channels = 4;
            int pad_size = 9; int stride = 3; int dilation = 2;
            int O_samples = I_samples; 
            int O_rows = (I_rows + 2 * pad_size - F_rows * dilation + dilation - 1) / stride + 1;
            int O_columns = (I_columns + 2 * pad_size - F_columns * dilation + dilation - 1) / stride + 1;
            int O_channels = F_num;
            
            Tensor I_Conv = new Tensor(4, I_samples, I_rows, I_columns, I_channels);
            Tensor T_Conv = new Tensor(4, O_samples, O_rows, O_columns, O_channels);
            for (int i=0; i < I_Conv.values.Length; i++) {I_Conv.values[i] = rand.NextDouble();}
            for (int i=0; i < T_Conv.values.Length; i++) {T_Conv.values[i] = rand.NextDouble();}

            Batch_Normalization_Layer BN = new Batch_Normalization_Layer(d);
            Convolution_Layer Conv = new Convolution_Layer(I_channels, F_num, F_rows, F_columns, true, pad_size, stride, dilation);
            Mean_Squared_Loss_Layer MSE = new Mean_Squared_Loss_Layer();
            Input_Layer Input = new Input_Layer();
            
            // Gradient Test
            Tuple<Tensor, Tensor, Tensor> analytic_gradients = analytic_grad(Conv, BN, MSE, I_Conv, T_Conv);
            Tuple<Tensor, Tensor, Tensor> numeric_gradients = numeric_grad(Conv, BN, MSE, I_Conv, T_Conv);

            Console.WriteLine("Difference in bias gradients\n" + Utils.sum(analytic_gradients.Item1.difference(numeric_gradients.Item1)) + "\n");
            Console.WriteLine("Difference in weight gradients\n" + Utils.sum(analytic_gradients.Item2.difference(numeric_gradients.Item2)) + "\n");
            Console.WriteLine("Difference in input gradients\n" + Utils.sum(analytic_gradients.Item3.difference(numeric_gradients.Item3)) + "\n");
        }
    }
}
