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
        public Mean_Squared_Loss MSE;

        public Tensor I, BN_I, T, BN_T, dO;

        public int I_samples, I_rows, I_columns, I_channels;
        public int n, d;

        public int F_num, F_rows, F_columns, F_channels;
        int pad_size;
        int stride;
        int dilation;
        int O_samples, O_rows, O_columns, O_channels;


        

        public test_CNN() {

            I_samples = 2; I_rows = 3; I_columns = 3; I_channels = 4;
            F_num = 3; F_rows = 2; F_columns = 2; F_channels = 4;
            
            pad_size = 0;
            stride = 1;
            dilation = 1;

            O_samples = I_samples;
            O_rows = (I_rows + 2 * pad_size - F_rows * dilation + dilation - 1) / stride + 1;
            O_columns = (I_columns + 2 * pad_size - F_columns * dilation + dilation - 1) / stride + 1;
            O_channels = F_num;


            n = 8;
            d = 3;

            // batch norm input tensor
            this.BN_I = new Tensor(2, n, d);
            Double[] input = {-1.1258, -1.1524, -0.2506,
                                -0.4339,  0.8487,  0.6920,
                                -0.3160, -2.1152,  0.4681,
                                -0.1577,  1.4437,  0.2660,
                                0.1665,  0.8744, -0.1435,
                                -0.1116,  0.9318,  1.2590,
                                2.0050,  0.0537,  0.6181,
                                -0.4128, -0.8411, -2.3160};
            this.BN_I.values = input;

            // batch norm target tensor
            this.BN_T = new Tensor(2, n, d);
            Double[] target = {-0.2159, -0.7425,  0.5627,
                                0.2596, -0.1740, -0.6787,
                                0.9383,  0.4889, -0.5692,
                                0.9200,  1.1108,  1.2899,
                                -1.4782,  2.5672, -0.4731,
                                0.3356, -1.6293, -0.5497,
                                -0.4798, -0.4997, -1.0670,
                                1.1149, -0.1407,  0.8058 };
            BN_T.values = target;

            this.dO = new Tensor(2, n, d);
            Double[] gradOut = {-0.0875, -0.0206, -0.0737,
                                -0.0594, 0.0746,  0.1076,
                                -0.1044, -0.1918, 0.08,
                                -0.0874, 0.0099,  -0.0916,
                                0.1442,  -0.152,  0.0214,
                                -0.0342, 0.2018,  0.1437,
                                0.2411,  0.0451,  0.1339,
                                -0.1286, -0.0486, -0.2646};
            dO.values = gradOut;

            // test input tensor
            //this.I = new Tensor(4, I_samples, I_rows, I_columns, I_channels);
            //for (int i = 0; i < I_samples; i++) {
            //    for (int j = 0; j < I_rows; j++) {
            //        for (int k = 0; k < I_columns; k++) {
            //            for (int l = 0; l < I_channels; l++) {
            //                I.values[I.index(i, j, k, l)] = I.index(i, j, k, l) / 10.0;
            //            }
            //        }
            //    }
            //}

            // test target tensor
            //this.T = new Tensor(4, O_samples, O_rows, O_columns, O_channels);
            //for (int i = 0; i < O_samples; i++) {
            //    for (int j = 0; j < O_rows; j++) {
            //        for (int k = 0; k < O_columns; k++) {
            //            for (int l = 0; l < O_channels; l++) {
            //                T.values[T.index(i, j, k, l)] = T.index(i, j, k, l) / 20.0;
            //            }
            //        }
            //    }
            //}

            // this.Conv = new Convolution_Layer(I_channels, F_num, F_rows, F_columns, true, pad_size, stride, dilation);
            this.BN = new Batch_Normalization_Layer(d);
            this.MSE = new Mean_Squared_Loss();

            // Set filters
            // test target tensor
            //for (int i = 0; i < F_num; i++) {
            //    for (int j = 0; j < F_rows; j++) {
            //        for (int k = 0; k < F_columns; k++) {
            //            for (int l = 0; l < F_channels; l++) {
            //                this.Conv.F.values[this.Conv.F.index(i, j, k, l)] = this.Conv.F.index(i, j, k, l) / 100.0;
            //            }
            //        }
            //    }
            //}


            //// Set biases
            //for (int i = 0; i < F_num; i++) {
            //    this.Conv.B.values[i] = (i + 1) / 10.0;
            //}
        }

        public Tensor forward(Tensor input) {
            
            // A = this.Conv.forward(this.I);
            Tensor I = new Tensor(input.dimensions, input.dim_1, input.dim_2, input.dim_3, input.dim_4);
            for (int k = 0; k < input.values.Count(); k++) {
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

        public static Tensor grad_check(Func<Tensor, bool, Tensor> forward, Tensor I, Tensor dO, Double h = 0.00001) {
            Tensor numeric_gradient = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);

            for (int i=0; i < I.values.Count(); i++) {
                
                Tensor I_up = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);
                Tensor I_down = new Tensor(I.dimensions, I.dim_1, I.dim_2, I.dim_3, I.dim_4);
                for (int j=0; j < I.values.Count(); j++) {
                    I_up.values[j] = I.values[j];
                    I_down.values[j] = I.values[j];
                }
                I_up.values[i] += h;
                Tensor up = (forward(I_up, true));
                I_down.values[i] -= h;
                Tensor down = forward(I_down, true);
                
                // 1. (f(x + h) - f (x - h)) * dL/dO (elementwise)
                // 2. Sum all elements
                // 3. Divide by (2 * h)
                numeric_gradient.values[i] = Utils.sum(Utils.elementwise_product(Utils.subtract(up, down), dO)) / (2 * h);
            }
            return numeric_gradient;
        }









        public static void test () {


            test_CNN test_CNN = new test_CNN();
            Double loss_up = 0.0;
            Double loss_down = 0.0;
            Double h = 0.00001;

            // Tensor analytic_dI;
            // Tensor analytic_dB;
            // Tensor analytic_dF;
            Tensor analytic_d_gamma;
            Tensor analytic_d_beta;
            Tensor analytic_d_I_BN;

            // Tensor numeric_dI;
            // Tensor numeric_dB;
            // Tensor numeric_dF;
            Tensor numeric_d_gamma;
            Tensor numeric_d_beta;
            Tensor numeric_d_I_BN = new Tensor (2, test_CNN.n, test_CNN.d);

            // Numerical gradient of loss with respect to input for BN
            for (int i = 0; i < test_CNN.n; i++) {
                for (int j = 0; j < test_CNN.d; j++) {
                    Tensor I_up = new Tensor(test_CNN.BN_I.dimensions, test_CNN.BN_I.dim_1, test_CNN.BN_I.dim_2, test_CNN.BN_I.dim_3, test_CNN.BN_I.dim_4);
                    Tensor I_down = new Tensor(test_CNN.BN_I.dimensions, test_CNN.BN_I.dim_1, test_CNN.BN_I.dim_2, test_CNN.BN_I.dim_3, test_CNN.BN_I.dim_4);
                    for (int k=0; k < test_CNN.BN_I.values.Count(); k++) {
                        I_up.values[k] = test_CNN.BN_I.values[k];
                        I_down.values[k] = test_CNN.BN_I.values[k];
                    }
                    I_up.values[i * test_CNN.d + j] += h;
                    I_down.values[i * test_CNN.d + j] -= h;



                    loss_up = 0.0;
                    loss_down = 0.0;

                    
                    loss_up += test_CNN.forward(I_up).values[0];
                    
                    loss_down += test_CNN.forward(I_down).values[0];
                    numeric_d_I_BN.values[i * test_CNN.d + j] = (loss_up - loss_down) / (2 * h);
                }
            }
            // Console.WriteLine("Analytic d_I:");
            //Console.WriteLine(analytic_d_I_BN);

            //Console.WriteLine("Numeric d_I:");
              Console.WriteLine(numeric_d_I_BN);

            // Console.WriteLine("Difference");
            //Console.WriteLine(analytic_d_I_BN.difference(numeric_d_I_BN));




            //numeric_d_I_BN = grad_check(test_CNN.BN.forward, test_CNN.BN_I, test_CNN.dO);

            //test_CNN.forward();


            //analytic_dI = test_CNN.backward();
            //analytic_dB = test_CNN.Conv.dB;
            //analytic_dF = test_CNN.Conv.dF;
            analytic_d_I_BN = test_CNN.backward();
            analytic_d_gamma = test_CNN.BN.d_gamma;
            analytic_d_beta = test_CNN.BN.d_beta;
            

            //Console.WriteLine(analytic_d_I_BN.difference(numeric_d_I_BN));


            //numeric_dI = new Tensor(analytic_dI.dimensions, analytic_dI.dim_1, analytic_dI.dim_2, analytic_dI.dim_3, analytic_dI.dim_4);
            //numeric_dB = new Tensor(analytic_dB.dimensions, analytic_dB.dim_1);
            //numeric_dF = new Tensor(analytic_dF.dimensions, analytic_dF.dim_1, analytic_dF.dim_2, analytic_dF.dim_3, analytic_dF.dim_4);
            //numeric_d_gamma = new Tensor(1, test_CNN.d);
            //numeric_d_beta = new Tensor(1, test_CNN.d);
            //numeric_d_I_BN = new Tensor(2, test_CNN.n, test_CNN.d);

            //int B_num = test_CNN.Conv.B.dim_1;
            //int F_num = test_CNN.Conv.F.dim_1; int F_rows = test_CNN.Conv.F.dim_2; int F_columns = test_CNN.Conv.F.dim_3; int F_channels = test_CNN.Conv.F.dim_4;
            //int I_samples = test_CNN.I_samples;

           


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

            //// Numerical gradient of loss with respect to input
            //for (int i = 0; i < test_CNN.I.dim_1; i++) {
            //    for (int j = 0; j < test_CNN.I.dim_2; j++) {
            //        for (int k = 0; k < test_CNN.I.dim_3; k++) {
            //            for (int l = 0; l < test_CNN.I.dim_4; l++) {
            //                loss_up = 0.0;
            //                loss_down = 0.0;

            //                test_CNN.I.values[test_CNN.I.index(i, j, k, l)] += h;
            //                for (int s = 0; s < test_CNN.forward().values.Count(); s++) {
            //                    loss_up = test_CNN.forward().values[i];
            //                }
            //                test_CNN.I.values[test_CNN.I.index(i, j, k, l)] -= 2 * h;
            //                for (int s = 0; s < test_CNN.forward().values.Count(); s++) {
            //                    loss_down = test_CNN.forward().values[i];
            //                }
            //                test_CNN.I.values[test_CNN.I.index(i, j, k, l)] += h;

            //                numeric_dI.values[numeric_dI.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h * test_CNN.I.dim_1 * test_CNN.I.dim_1);
            //            }
            //        }
            //    }
            //}
            //Console.WriteLine("--------------------------------------");
            //Console.WriteLine("ANALYTIC DI");
            //Console.WriteLine(analytic_dI);
            //Console.WriteLine(numeric_dI);
            //Console.WriteLine(analytic_dI.difference(numeric_dI));
        }
    }
}
