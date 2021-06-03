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

        public Tensor I, BN_I, T, BN_T;

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

        public Tensor forward() {
            Tensor A;
            // A = this.Conv.forward(this.I);
            A = this.BN.forward(this.BN_I, true);
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

        public static void test () {
            
            Double loss_up = 0.0;
            Double loss_down = 0.0;
            Double h = 0.0000001;

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
            Tensor numeric_d_I_BN;

            test_CNN test_CNN = new test_CNN();

            test_CNN.forward();


            //analytic_dI = test_CNN.backward();
            //analytic_dB = test_CNN.Conv.dB;
            //analytic_dF = test_CNN.Conv.dF;
            analytic_d_I_BN = test_CNN.backward();
            analytic_d_gamma = test_CNN.BN.d_gamma;
            analytic_d_beta = test_CNN.BN.d_beta;

            //numeric_dI = new Tensor(analytic_dI.dimensions, analytic_dI.dim_1, analytic_dI.dim_2, analytic_dI.dim_3, analytic_dI.dim_4);
            //numeric_dB = new Tensor(analytic_dB.dimensions, analytic_dB.dim_1);
            //numeric_dF = new Tensor(analytic_dF.dimensions, analytic_dF.dim_1, analytic_dF.dim_2, analytic_dF.dim_3, analytic_dF.dim_4);
            //numeric_d_gamma = new Tensor(1, test_CNN.d);
            //numeric_d_beta = new Tensor(1, test_CNN.d);
            //numeric_d_I_BN = new Tensor(2, test_CNN.n, test_CNN.d);

            //int B_num = test_CNN.Conv.B.dim_1;
            //int F_num = test_CNN.Conv.F.dim_1; int F_rows = test_CNN.Conv.F.dim_2; int F_columns = test_CNN.Conv.F.dim_3; int F_channels = test_CNN.Conv.F.dim_4;
            //int I_samples = test_CNN.I_samples;

            // Numerical gradient of loss with respect to input for BN
            //for (int i = 0; i < test_CNN.n; i++) {
            //    for (int j = 0; j < test_CNN.d; j++) {
            //        loss_up = 0.0;
            //        loss_down = 0.0;

            //        test_CNN.BN_I.values[i * test_CNN.d + j] += h;
            //        for (int k = 0; k < test_CNN.n; k++) {
            //            loss_up += test_CNN.forward().values[k];
            //        }
            //        test_CNN.BN_I.values[i * test_CNN.d + j] -= 2 * h;
            //        for (int k = 0; k < test_CNN.n; k++) {
            //            loss_down += test_CNN.forward().values[k];
            //        }
            //        test_CNN.BN_I.values[i * test_CNN.d + j] += h;

            //        numeric_d_I_BN.values[i * test_CNN.d + j] = (loss_up - loss_down) / (2 * h * test_CNN.n);
            //    }
            //}
            Console.WriteLine("Analytic d_I:");
            Console.WriteLine(analytic_d_I_BN);

            //Console.WriteLine("Numeric d_I:");
            //Console.WriteLine(numeric_d_I_BN);

            //Console.WriteLine("Difference");
            //Console.WriteLine(analytic_d_I_BN.difference(numeric_d_I_BN));


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
            Console.WriteLine("Analytic d_gamma:");
            Console.WriteLine(analytic_d_gamma);

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
            Console.WriteLine("Analytic d_beta:");
            Console.WriteLine(analytic_d_beta);

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
