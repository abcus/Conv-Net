using System;

namespace Conv_Net {

    class test_CNN {

        public Convolution_Layer Conv; 
        public Mean_Squared_Loss MSE;
        public Tensor I, T;

        public test_CNN() {

            // Input: 5 rows x 5 columns x 2 channels
            // Padding: 4
            // Filters: 2 num x 3 rows x 3 columns x 2 channels
            // Dilation: 3
            // Stride 2
            // Output size: 4 rows x 4 columns x 2 channels

            // test input tensor
            this.I = new Tensor(4, 1, 5, 5, 2);
            for (int i = 0; i < I.dim_1; i++) {
                for (int j = 0; j < I.dim_2; j++) {
                    for (int k = 0; k < I.dim_3; k++) {
                        for (int l=0; l < I.dim_4; l++) {
                            this.I.values[this.I.index(i, j, k, l)] = ((j + k + 1) * (l + 1) / 10.0);
                        }
                    }
                }
            }

            // test target tensor
            this.T = new Tensor(4, 1, 4, 4, 2);
            T.values[0] = 4.7; T.values[1] = 8.4; 
            T.values[2] = 3.8; T.values[3] = 7.2;
            T.values[4] = 5.2; T.values[5] = 9.3;
            T.values[6] = 3.1; T.values[7] = 5.4;
            T.values[8] = 4.1; T.values[9] = 3.2;
            T.values[10] = 4.4; T.values[11] = 6.2;
            T.values[12] = 1.3; T.values[13] = 2.6;
            T.values[14] = 5.7; T.values[15] = 3.2;
            T.values[16] = 4.4; T.values[17] = 2.1;
            T.values[18] = 1.2; T.values[19] = 1.8;
            T.values[20] = 2.8; T.values[21] = 4.3;
            T.values[22] = 2.5; T.values[23] = 5.7;
            T.values[24] = 3.5; T.values[25] = 4.3;
            T.values[26] = 5.5; T.values[27] = 7.3;
            T.values[28] = 8.9; T.values[29] = 3.4;
            T.values[30] = 1.0; T.values[31] = 0.4;

            this.Conv = new Convolution_Layer(2, 2, 3, 3, true, 4, 2, 3);
            this.MSE = new Mean_Squared_Loss();
        }

        public Tensor forward() {
            Tensor A; 
            A = this.Conv.forward(this.I);
            A = this.MSE.forward(A, this.T);
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
            analytic_dB = test_CNN.Conv.dB; analytic_dB.dimensions = 1; analytic_dB.dim_1 = analytic_dB.dim_2; analytic_dB.dim_2 = 1;
            analytic_dF = test_CNN.Conv.dF; analytic_dF.dimensions = 4; analytic_dF.dim_1 = analytic_dF.dim_2; analytic_dF.dim_2 = analytic_dF.dim_3; analytic_dF.dim_3 = analytic_dF.dim_4; analytic_dF.dim_4 = analytic_dF.dim_5; analytic_dF.dim_5 = 1;

            numeric_dI = new Tensor(analytic_dI.dimensions, analytic_dI.dim_1, analytic_dI.dim_2, analytic_dI.dim_3, analytic_dI.dim_4);
            numeric_dB = new Tensor(analytic_dB.dimensions, analytic_dB.dim_1);
            numeric_dF = new Tensor(analytic_dF.dimensions, analytic_dF.dim_1, analytic_dF.dim_2, analytic_dF.dim_3, analytic_dF.dim_4);

            //Numerical gradient of loss with respect to bias
            for (int i = 0; i < test_CNN.Conv.B.dim_1; i++) {
                loss_up = 0.0;
                loss_down = 0.0;

                test_CNN.Conv.B.values[i] += h;
                loss_up = test_CNN.forward().values[0];
                test_CNN.Conv.B.values[i] -= 2 * h;
                loss_down = test_CNN.forward().values[0];
                test_CNN.Conv.B.values[i] += h;

                numeric_dB.values[i] = (loss_up - loss_down) / (2 * h);
            }
            // Console.WriteLine(analytic_dB);
            // Console.WriteLine(numeric_dB);
            Console.WriteLine(analytic_dB.difference(numeric_dB));

            // Numerical gradient of loss with respect to filters
            for (int i = 0; i < test_CNN.Conv.F.dim_1; i++) {
                for (int j = 0; j < test_CNN.Conv.F.dim_2; j++) {
                    for (int k = 0; k < test_CNN.Conv.F.dim_3; k++) {
                        for (int l = 0; l < test_CNN.Conv.F.dim_4; l++) {
                            loss_up = 0.0;
                            loss_down = 0.0;

                            test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] += h;
                            loss_up = test_CNN.forward().values[0];
                            test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] -= 2 * h;
                            loss_down = test_CNN.forward().values[0];
                            test_CNN.Conv.F.values[test_CNN.Conv.F.index(i, j, k, l)] += h;

                            numeric_dF.values[numeric_dF.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h);
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
                            loss_up = test_CNN.forward().values[0];
                            test_CNN.I.values[test_CNN.I.index(i, j, k, l)] -= 2 * h;
                            loss_down = test_CNN.forward().values[0];
                            test_CNN.I.values[test_CNN.I.index(i, j, k, l)] += h;

                            numeric_dI.values[numeric_dI.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h);
                        }
                    }
                }
            }
            //Console.WriteLine(analytic_dI);
            //Console.WriteLine(numeric_dI);
            Console.WriteLine(analytic_dI.difference(numeric_dI));
        }
    }
}
