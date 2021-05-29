using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {

    class test_CNN {

        public Convolution_Layer Conv; 
        public Mean_Squared_Loss MSE;
        public Tensor I, T;

        public test_CNN() {

            // Input: 2 samples x 5 rows x 5 columns x 2 channels
            // Padding: 4
            // Filters: 2 num x 3 rows x 3 columns x 2 channels
            // Dilation: 3
            // Stride 2
            // Output size: 2 samples x 4 rows x 4 columns x 2 channels

            // test input tensor
            this.I = new Tensor(4, 2, 5, 5, 2);
            for (int i = 0; i < I.dim_1 * I.dim_2 * I.dim_3 * I.dim_4; i++) {
                this.I.values[i] = i / 100.0;
            }

            // test target tensor
            this.T = new Tensor(4, 2, 4, 4, 2);
            for (int i=0; i < T.dim_1 * T.dim_2 * T.dim_3 * T.dim_4; i++) {
                this.T.values[i] = i / 20.0;
            }

            this.Conv = new Convolution_Layer(2, 2, 3, 3, true, 4, 2, 3);
            this.MSE = new Mean_Squared_Loss();

            for (int i=0; i < this.Conv.F_num * this.Conv.F_rows * this.Conv.F_columns * this.Conv.F_channels; i++) {
                this.Conv.F.values[i] = i / 100.0;
            }
        }

        public Tensor forward() {
            Tensor A; 
            A = this.Conv.forward(this.I);
            A = this.MSE.loss(A, this.T);
            
            return A;
        }

        public Tensor backward() {
            Tensor Z;
            Z = this.MSE.backward();
            Z = Conv.backward(Z);
            return Z;
        }
    }
    
    static class Grad_Check {

        public static void test () {
            
            Double loss_up = 0.0;
            Double loss_down = 0.0;
            Double h = 0.0000001;

            Tensor analytic_dI;
            Tensor analytic_dB;
            Tensor analytic_dF;

            Tensor numeric_dI;
            Tensor numeric_dB;
            Tensor numeric_dF;

            test_CNN test_CNN = new test_CNN();

            test_CNN.forward();
            
            analytic_dI = test_CNN.backward();
            analytic_dB = new Tensor(1, test_CNN.Conv.dB.dim_2);
            analytic_dF = new Tensor(4, test_CNN.Conv.dF.dim_2, test_CNN.Conv.dF.dim_3, test_CNN.Conv.dF.dim_4, test_CNN.Conv.dF.dim_5);

            numeric_dI = new Tensor(analytic_dI.dimensions, analytic_dI.dim_1, analytic_dI.dim_2, analytic_dI.dim_3, analytic_dI.dim_4);
            numeric_dB = new Tensor(analytic_dB.dimensions, analytic_dB.dim_1);
            numeric_dF = new Tensor(analytic_dF.dimensions, analytic_dF.dim_1, analytic_dF.dim_2, analytic_dF.dim_3, analytic_dF.dim_4);

            int I_samples = test_CNN.I.dim_1;
            int B_num = test_CNN.Conv.B.dim_1;
            int F_num = test_CNN.Conv.F.dim_1; int F_rows = test_CNN.Conv.F.dim_2; int F_columns = test_CNN.Conv.F.dim_3; int F_channels = test_CNN.Conv.F.dim_4;

            // Analytic gradient of loss with respect to bias
            // For each bias, sum contribution from each sample (which has already been divided by batch size)
            for (int i=0; i < B_num; i++) {
                for (int s=0; s < I_samples; s++) {
                    analytic_dB.values[i] += test_CNN.Conv.dB.values[s * test_CNN.Conv.B.dim_1 + i];
                }
            }

            //Numerical gradient of loss with respect to bias
            // For each bias, sum contribution from each sample (divide by batch size at the end)
            for (int i = 0; i < B_num; i++) {
                loss_up = 0.0;
                loss_down = 0.0;

                test_CNN.Conv.B.values[i] += h;
                for (int s=0; s < I_samples; s++) {
                    loss_up += test_CNN.forward().values[s];
                }
                test_CNN.Conv.B.values[i] -= 2 * h;
                for (int s=0; s < I_samples; s++) {
                    loss_down += test_CNN.forward().values[s];
                }
                test_CNN.Conv.B.values[i] += h;
                numeric_dB.values[i] = (loss_up - loss_down) / (2 * h * I_samples);
            }
            // Console.WriteLine(analytic_dB);
            // Console.WriteLine(numeric_dB);
            Console.WriteLine(analytic_dB.difference(numeric_dB));

            // Analytic gradient of loss with respect to filters
            // For each filter, sum contributions from each sample (which has already been divided by batch size)
            for (int i=0; i < F_num; i++) {
                for (int j=0; j < F_rows; j++) {
                    for (int k=0; k < F_columns; k++) {
                        for (int l=0; l < F_channels; l++) {
                            for (int s = 0; s < I_samples; s++) {
                                analytic_dF.values[analytic_dF.index(i, j, k, l)] += test_CNN.Conv.dF.values[test_CNN.Conv.dF.index(s, i, j, k, l)];
                            }
                        }
                    }
                }
            }
            // Numerical gradient of loss with respect to filters
            // For each bias, sum contribution from each sample (divide by batch size at the end)
            for (int i = 0; i < test_CNN.Conv.F.dim_1; i++) {
                for (int j = 0; j < test_CNN.Conv.F.dim_2; j++) {
                    for (int k = 0; k < test_CNN.Conv.F.dim_3; k++) {
                        for (int l = 0; l < test_CNN.Conv.F.dim_4; l++) {
                            loss_up = 0.0;
                            loss_down = 0.0;

                            test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] += h;
                            for (int s = 0; s < I_samples; s++) {
                                loss_up += test_CNN.forward().values[s];
                            }
                            test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] -= 2 * h;
                            for (int s = 0; s < I_samples; s++) {
                                loss_down += test_CNN.forward().values[s];
                            }
                            test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] += h;
                            numeric_dF.values[numeric_dF.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h * test_CNN.I.dim_1);
                        }
                    }
                }
            }
            // Console.WriteLine(analytic_dF);
            // Console.WriteLine(numeric_dF);
            Console.WriteLine(analytic_dF.difference(numeric_dF));

            // Numerical gradient of loss with respect to input
            for (int i = 0; i < test_CNN.I.dim_1; i++) {
                for (int j = 0; j < test_CNN.I.dim_2; j++) {
                    for (int k = 0; k < test_CNN.I.dim_3; k++) {
                        for (int l = 0; l < test_CNN.I.dim_4; l++) {
                            loss_up = 0.0;
                            loss_down = 0.0;

                            test_CNN.I.values[test_CNN.I.index(i, j, k, l)] += h;
                            loss_up = test_CNN.forward().values[i];
                            test_CNN.I.values[test_CNN.I.index(i, j, k, l)] -= 2 * h;
                            loss_down = test_CNN.forward().values[i];
                            test_CNN.I.values[test_CNN.I.index(i, j, k, l)] += h;

                            numeric_dI.values[numeric_dI.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h * test_CNN.I.dim_1);
                        }
                    }
                }
            }
            // Console.WriteLine(analytic_dI);
            // Console.WriteLine(numeric_dI);
            Console.WriteLine(analytic_dI.difference(numeric_dI));
        }
    }
}
