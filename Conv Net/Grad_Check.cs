using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Conv_Net {

    class test_CNN {

        public Convolution_Layer Conv;
        public Batch_Normalization_Layer BN;
        public Fully_Connected_Layer FC;
        public Mean_Squared_Loss_Layer MSE;

        public Tensor I, BN_I, T, BN_T, dO;

        public int I_samples, I_rows, I_columns, I_channels;
        public int n, d;

        public int F_num, F_rows, F_columns, F_channels;
        int pad_size;
        int stride;
        int dilation;
        int O_samples, O_rows, O_columns, O_channels;


        

        public test_CNN() {




          

           

            // this.Conv = new Convolution_Layer(I_channels, F_num, F_rows, F_columns, true, pad_size, stride, dilation);
            // this.BN = new Batch_Normalization_Layer(d);
            // this.MSE = new Mean_Squared_Loss_Layer();

        }

        public Tensor forward(Tensor input) {
            
            // A = this.Conv.forward(this.I);
            Tensor I = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);
            for (int k = 0; k < input.values.Length; k++) {
                I.values[k] = input.values[k];
            }

            Tensor A;
            A = this.BN.forward(I, true);
            A = this.MSE.loss(A, this.BN_T);
            return A;
        }

        public Tensor backward() {
            Tensor Z;
            Z = this.MSE.backward();
            Z = this.BN.backward(Z);
            
            // Z = Conv.backward(Z);
            return Z;
        }
    }
    
    static class Grad_Check {

        /// <summary>
        /// Grad check based on code from CS 231n
        /// </summary>
        /// <param name="forward"> delegate for forward </param>
        /// <param name="I"></param>
        /// <param name="dO"></param>
        /// <param name="h"></param>
        /// <returns></returns>
        public static Tensor numeric_grad_1(Func<Tensor, bool, Tensor> forward, Tensor I, Tensor dO, Double h = 0.00001) {
            Tensor numeric_gradient = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);

            // For each element of the input, make two copies of the input tensor
            // Increment that element up by h in one copy, and down by h in the other copy
            for (int i=0; i < I.values.Length; i++) {
                Tensor I_up = Utils.copy(I);
                Tensor I_down = Utils.copy(I);

                I_up.values[i] += h;
                I_down.values[i] -= h;
                
                Tensor O_up = (forward(I_up, true));
                Tensor O_down = forward(I_down, true);
                
                // 1. ((forward(up) - forward(down)) * dO) (elementwise)
                // 2. Sum all elements
                // 3. Divide by (2 * h)
                numeric_gradient.values[i] = Utils.sum(Utils.elementwise_product(Utils.subtract(O_up, O_down), dO)) / (2 * h);
            }
            return numeric_gradient;
        }

        public static Tensor numeric_grad_2(Func<Tensor, bool, Tensor> forward, Func<Tensor, Tensor, Tensor> loss, Tensor I, Tensor T, Double h = 0.00001) {
            Tensor numeric_gradient = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);

            for (int i=0; i < I.values.Length; i++) {
                Tensor I_up = Utils.copy(I);
                Tensor I_down = Utils.copy(I);

                I_up.values[i] += h;
                I_down.values[i] -= h;

                Tensor L_up = loss(forward(I_up, true), T);
                Tensor L_down = loss(forward(I_down, true), T);

                numeric_gradient.values[i] = Utils.sum(Utils.subtract(L_up, L_down)) / (2 * h);
            }
            return numeric_gradient;
        }

        public static Tensor analytic_grad(Func<Tensor, bool, Tensor> layer_forward, Func<Tensor, Tensor, Tensor> loss_forward, Func<Tensor> loss_backwards, Func<Tensor, Tensor> layer_backwards, Tensor I, Tensor T) {
            Tensor I_copy = Utils.copy(I);
            Tensor analytic_dI;

            loss_forward(layer_forward(I_copy, true), T);
            analytic_dI = layer_backwards(loss_backwards());

            return analytic_dI;
        }


        public static Tensor numeric_grad_conv_BN(Func<Tensor, bool, Tensor> forward_conv, Func<Tensor, bool, Tensor> forward_BN, Func<Tensor, Tensor, Tensor> loss, Tensor I, Tensor T, Double h = 0.00001) {
            Tensor numeric_gradient = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);

            for (int i = 0; i < I.values.Length; i++) {
                Tensor I_up = Utils.copy(I);
                Tensor I_down = Utils.copy(I);

                I_up.values[i] += h;
                I_down.values[i] -= h;

                Tensor L_up = loss(forward_BN(forward_conv(I_up, true), true), T);
                Tensor L_down = loss(forward_BN(forward_conv(I_down, true), true), T);

                numeric_gradient.values[i] = Utils.sum(Utils.subtract(L_up, L_down)) / (2 * h);
            }
            return numeric_gradient;
        }

        
        public static Tensor analytic_grad_conv_BN(Convolution_Layer Conv, Batch_Normalization_Layer BN, Mean_Squared_Loss_Layer MSE, Tensor I, Tensor T) {
            Tensor I_copy = Utils.copy(I);
            Tensor analytic_dI;

            MSE.loss(BN.forward(Conv.forward(I_copy), true), T);
            analytic_dI = Conv.backward(BN.backward(MSE.backward()));

            return analytic_dI;
        }

        public static void test () {

            Random rand = new Random(0);

            // Batch norm test
            int n = 8; int d = 3;
            Tensor I_BN = new Tensor(2, n, d);
            Tensor T_BN = new Tensor(2, n, d);
            Tensor dO_BN = new Tensor(2, n, d);

            // Conv test
            int I_samples = 2; int I_rows = 3; int I_columns = 3; int I_channels = 4;
            int F_num = 3; int F_rows = 2; int F_columns = 2; int F_channels = 4;
            int pad_size = 0; int stride = 1; int dilation = 1;
            int O_samples = I_samples; 
            int O_rows = (I_rows + 2 * pad_size - F_rows * dilation + dilation - 1) / stride + 1;
            int O_columns = (I_columns + 2 * pad_size - F_columns * dilation + dilation - 1) / stride + 1;
            int O_channels = F_num;
            Tensor I_Conv = new Tensor(4, I_samples, I_rows, I_columns, I_channels);
            Tensor T_Conv = new Tensor(4, O_samples, O_rows, O_columns, O_channels);

            // batch norm input tensor
            Double[] temp = {-1.1258, -1.1524, -0.2506,
                                -0.4339,  0.8487,  0.6920,
                                -0.3160, -2.1152,  0.4681,
                                -0.1577,  1.4437,  0.2660,
                                0.1665,  0.8744, -0.1435,
                                -0.1116,  0.9318,  1.2590,
                                2.0050,  0.0537,  0.6181,
                                -0.4128, -0.8411, -2.3160};
            I_BN.values = temp;

            // batch norm target tensor
            Double[] temp1 = {-0.2159, -0.7425,  0.5627,
                                0.2596, -0.1740, -0.6787,
                                0.9383,  0.4889, -0.5692,
                                0.9200,  1.1108,  1.2899,
                                -1.4782,  2.5672, -0.4731,
                                0.3356, -1.6293, -0.5497,
                                -0.4798, -0.4997, -1.0670,
                                1.1149, -0.1407,  0.8058 };
            T_BN.values = temp1;

            Double[] temp2= {-0.0875, -0.0206, -0.0737,
                                -0.0594, 0.0746,  0.1076,
                                -0.1044, -0.1918, 0.08,
                                -0.0874, 0.0099,  -0.0916,
                                0.1442,  -0.152,  0.0214,
                                -0.0342, 0.2018,  0.1437,
                                0.2411,  0.0451,  0.1339,
                                -0.1286, -0.0486, -0.2646};
            dO_BN.values = temp2;

            // Conv input tensor
            for (int i=0; i < I_Conv.values.Length; i++) {
                I_Conv.values[i] = rand.NextDouble();
            }

            // Conv target tensor
            for (int i=0; i < T_Conv.values.Length; i++) {
                T_Conv.values[i] = rand.NextDouble();
            }

            Batch_Normalization_Layer BN = new Batch_Normalization_Layer(d);
            Convolution_Layer Conv = new Convolution_Layer(I_channels, F_num, F_rows, F_columns, true, pad_size, stride, dilation);
            Mean_Squared_Loss_Layer MSE = new Mean_Squared_Loss_Layer();


            // Batch norm analytic and numeric gradients
            // Tensor analytic_dI_BN = analytic_grad(BN.forward, MSE.loss, MSE.backward, BN.backward, I_BN, T_BN);
            // Tensor numeric_dI_BN1 = numeric_grad_1(BN.forward, I_BN, dO_BN);
            // Tensor numeric_dI_BN2 = numeric_grad_2(BN.forward, MSE.loss, I_BN, T_BN);
            // Console.WriteLine(analytic_dI_BN.difference(numeric_dI_BN1)); // Larger error due to precion with dO values
            // Console.WriteLine(analytic_dI_BN.difference(numeric_dI_BN2));

            // Conv analytic and numeric gradients
            Tensor analytic_dI_conv = analytic_grad(Conv.forward, MSE.loss, MSE.backward, Conv.backward, I_Conv, T_Conv);
            Tensor numeric_dI_conv = numeric_grad_2(Conv.forward, MSE.loss, I_Conv, T_Conv);
            // Console.WriteLine(analytic_dI_conv);
            // Console.WriteLine(numeric_dI_conv);
            Console.WriteLine(analytic_dI_conv.difference(numeric_dI_conv));

            // Conv + BN analytic and numeric gradients
            // Tensor analytic_dI_conv_BN = analytic_grad_conv_BN(Conv, BN, MSE, I_Conv, T_Conv);
            // Tensor numeric_dI_conv_BN = numeric_grad_conv_BN(Conv.forward, BN.forward, MSE.loss, I_Conv, T_Conv);
            // Console.WriteLine(analytic_dI_conv_BN);
            // Console.WriteLine(numeric_dI_conv_BN);
            // Console.WriteLine(numeric_dI_conv_BN.difference(analytic_dI_conv_BN));



            //Double loss_up = 0.0;
            //Double loss_down = 0.0;
            //Double h = 0.00001;

            // Tensor analytic_dI;
            // Tensor analytic_dB;
            // Tensor analytic_dF;
            //Tensor analytic_d_gamma;
            //Tensor analytic_d_beta;
            //Tensor analytic_d_I_BN;

            // Tensor numeric_dI;
            // Tensor numeric_dB;
            // Tensor numeric_dF;
            //Tensor numeric_d_gamma;
            //Tensor numeric_d_beta;
            //Tensor numeric_d_I_BN = new Tensor (2, test_CNN.n, test_CNN.d);

            // numeric_d_I_BN = grad_check(test_CNN.BN.forward, test_CNN.BN_I, test_CNN.dO);
            // Console.WriteLine(numeric_d_I_BN);

           



            //analytic_dI = test_CNN.backward();
            //analytic_dB = test_CNN.Conv.dB;
            //analytic_dF = test_CNN.Conv.dF;
            //analytic_d_I_BN = test_CNN.backward();
            //analytic_d_gamma = test_CNN.BN.d_gamma;
            //analytic_d_beta = test_CNN.BN.d_beta;
            

            //Console.WriteLine(analytic_d_I_BN.difference(numeric_d_I_BN));


            //numeric_dI = new Tensor(analytic_dI.dimensions, analytic_dI.dim_1, analytic_dI.dim_2, analytic_dI.dim_3, analytic_dI.dim_4);
            //numeric_dB = new Tensor(analytic_dB.dimensions, analytic_dB.dim_1);
            //numeric_dF = new Tensor(analytic_dF.dimensions, analytic_dF.dim_1, analytic_dF.dim_2, analytic_dF.dim_3, analytic_dF.dim_4);
            //numeric_d_gamma = new Tensor(1, test_CNN.d);
            //numeric_d_beta = new Tensor(1, test_CNN.d);
            //numeric_d_I_BN = new Tensor(2, test_CNN.n, test_CNN.d);



           


            // Numerical gradient of loss with respect to gamma
            //for (int i = 0; i < test_CNN.d; i++) {

            //    loss_up = 0.0;
            //    loss_down = 0.0;

            //    test_CNN.BN.gamma.values[i] += h;
            //    for (int s = 0; s < test_CNN.n; s++) {
            //        loss_up += test_CNN.forward().values[s];
            //    }
            //    test_CNN.BN.gamma.values[i] -= 2 * h;
            //    for (int s = 0; s < test_CNN.n; s++) {
            //        loss_down += test_CNN.forward().values[s];
            //    }
            //    test_CNN.BN.gamma.values[i] += h;
            //    numeric_d_gamma.values[i] = (loss_up - loss_down) / (2 * h * test_CNN.n);
            //}
            // Console.WriteLine("Analytic d_gamma:");
            // Console.WriteLine(analytic_d_gamma);

            //Console.WriteLine("Numeric d_gamma:");
            //Console.WriteLine(numeric_d_gamma);

            //Console.WriteLine("Difference");
            //Console.WriteLine(analytic_d_gamma.difference(numeric_d_gamma));



            // Numerical gradient of loss with respect to beta
            //for (int i = 0; i < test_CNN.d; i++) {

            //    loss_up = 0.0;
            //    loss_down = 0.0;

            //    test_CNN.BN.beta.values[i] += h;
            //    for (int s = 0; s < test_CNN.n; s++) {
            //        loss_up += test_CNN.forward().values[s];
            //    }
            //    test_CNN.BN.beta.values[i] -= 2 * h;
            //    for (int s = 0; s < test_CNN.n; s++) {
            //        loss_down += test_CNN.forward().values[s];
            //    }
            //    test_CNN.BN.beta.values[i] += h;
            //    numeric_d_beta.values[i] = (loss_up - loss_down) / (2 * h * test_CNN.n);
            //}
            // Console.WriteLine("Analytic d_beta:");
            // Console.WriteLine(analytic_d_beta);

            //Console.WriteLine("Numeric d_beta:");
            //Console.WriteLine(numeric_d_beta);

            //Console.WriteLine("difference:");
            //Console.WriteLine(analytic_d_beta.difference(numeric_d_beta));


            ////Numerical gradient of loss with respect to bias
            //// For each bias, sum contribution from each sample (divide by batch size at the end)
            //for (int i = 0; i < B_num; i++) {
            //    loss_up = 0.0;
            //    loss_down = 0.0;

            //    test_CNN.Conv.B.values[i] += h;
            //    for (int s=0; s < test_CNN.forward().values.Count(); s++) {
            //        loss_up += test_CNN.forward().values[s];
            //    }
            //    test_CNN.Conv.B.values[i] -= 2 * h;
            //    for (int s = 0; s < test_CNN.forward().values.Count(); s++) {
            //        loss_down += test_CNN.forward().values[s];
            //    }
            //    test_CNN.Conv.B.values[i] += h;
            //    numeric_dB.values[i] = (loss_up - loss_down) / (2 * h * I_samples);
            //}
            //Console.WriteLine("--------------------------------------");
            //Console.WriteLine("ANALYTIC DB");
            //Console.WriteLine(analytic_dB);
            //Console.WriteLine(numeric_dB);
            //Console.WriteLine(analytic_dB.difference(numeric_dB));

            //// Numerical gradient of loss with respect to filters
            //// For each bias, sum contribution from each sample (divide by batch size at the end)
            //for (int i = 0; i < test_CNN.Conv.F.dim_1; i++) {
            //    for (int j = 0; j < test_CNN.Conv.F.dim_2; j++) {
            //        for (int k = 0; k < test_CNN.Conv.F.dim_3; k++) {
            //            for (int l = 0; l < test_CNN.Conv.F.dim_4; l++) {
            //                loss_up = 0.0;
            //                loss_down = 0.0;

            //                test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] += h;
            //                for (int s = 0; s < test_CNN.forward().values.Count(); s++) {
            //                    loss_up += test_CNN.forward().values[s];
            //                }
            //                test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] -= 2 * h;
            //                for (int s = 0; s < test_CNN.forward().values.Count(); s++) {
            //                    loss_down += test_CNN.forward().values[s];
            //                }
            //                test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] += h;
            //                numeric_dF.values[numeric_dF.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h * test_CNN.I.dim_1);
            //            }
            //        }
            //    }
            //}
            //Console.WriteLine("--------------------------------------");
            //Console.WriteLine("ANALYTIC DF");
            //Console.WriteLine(analytic_dF);
            //Console.WriteLine(numeric_dF);
            //Console.WriteLine(analytic_dF.difference(numeric_dF));

          
        }
    }
}
