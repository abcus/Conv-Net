﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {

    class test_gemm {

        public Convolution_Layer Conv;
        public Mean_Squared_Loss MSE;
        public Tensor I, T;

        public test_gemm() {

            // Input: 1 sample x 5 rows x 5 columns x 2 channels
            // Padding: 4
            // Filters: 2 num x 3 rows x 3 columns x 2 channels
            // Dilation: 3
            // Stride 2
            // Output size: 1 sample x 4 rows x 4 columns x 2 channels

            // test input tensor
            this.I = new Tensor(4, 1, 5, 5, 2);
            for (int i = 0; i < I.dim_1; i++) {
                for (int j = 0; j < I.dim_2; j++) {
                    for (int k = 0; k < I.dim_3; k++) {
                        for (int l = 0; l < I.dim_4; l++) {
                            this.I.values[this.I.index(i, j, k, l)] = ((j + k + 1) * (l + 1) / 10.0);
                        }
                    }
                }
            }

            // test target tensor
            this.T = new Tensor(4, 2, 4, 4, 2);
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

            T.values[32] = 4.1; T.values[33] = 8.1;
            T.values[34] = 3.2; T.values[35] = 7.2;
            T.values[36] = 5.3; T.values[37] = 9.4;
            T.values[38] = 3.4; T.values[39] = 5.3;
            T.values[40] = 4.5; T.values[41] = 3.4;
            T.values[42] = 4.6; T.values[43] = 6.5;
            T.values[44] = 1.7; T.values[45] = 2.7;
            T.values[46] = 5.8; T.values[47] = 3.6;
            T.values[48] = 4.9; T.values[49] = 2.7;
            T.values[50] = 1.1; T.values[51] = 1.9;
            T.values[52] = 2.2; T.values[53] = 4.0;
            T.values[54] = 2.3; T.values[55] = 5.1;
            T.values[56] = 3.4; T.values[57] = 4.2;
            T.values[58] = 5.5; T.values[59] = 7.5;
            T.values[60] = 8.6; T.values[61] = 3.7;
            T.values[62] = 1.7; T.values[63] = 0.9;

            int filter_nums = 2;
            int filter_rows = 3;
            int filter_columns = 3;
            int filter_channels = 2;
            int padding = 4;
            int dilation = 3;
            int stride = 2;

            this.Conv = new Convolution_Layer(filter_channels, filter_nums, filter_rows, filter_columns, true, padding, stride, dilation);
            this.MSE = new Mean_Squared_Loss();

           Tensor FF = Conv.F.F_2_col();
            // Console.WriteLine(FF);
            I = I.pad(padding);
            //Console.WriteLine(I);
            I = I.im_2_col(filter_rows, filter_columns, filter_channels, dilation, stride);
            // Console.WriteLine(I);
            Tensor Result = FF.mm(I);
            Console.WriteLine(Result);
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

    static class Gemm_Check {

        public static void test() {

            Double loss_up = 0.0;
            Double loss_down = 0.0;
            Double h = 0.0000001;

            Tensor analytic_dI;
            Tensor analytic_dB;
            Tensor analytic_dF;

            Tensor numeric_dI;
            Tensor numeric_dB;
            Tensor numeric_dF;

            test_gemm test_net = new test_gemm();



  

            //test_net.forward();

            //analytic_dI = test_net.backward();
            //analytic_dB = test_net.Conv.dB; analytic_dB.dimensions = 1; analytic_dB.dim_1 = analytic_dB.dim_2; analytic_dB.dim_2 = 1;
            //analytic_dF = test_net.Conv.dF; analytic_dF.dimensions = 4; analytic_dF.dim_1 = analytic_dF.dim_2; analytic_dF.dim_2 = analytic_dF.dim_3; analytic_dF.dim_3 = analytic_dF.dim_4; analytic_dF.dim_4 = analytic_dF.dim_5; analytic_dF.dim_5 = 1;

            //numeric_dI = new Tensor(analytic_dI.dimensions, analytic_dI.dim_1, analytic_dI.dim_2, analytic_dI.dim_3, analytic_dI.dim_4);
            //numeric_dB = new Tensor(analytic_dB.dimensions, analytic_dB.dim_1);
            //numeric_dF = new Tensor(analytic_dF.dimensions, analytic_dF.dim_1, analytic_dF.dim_2, analytic_dF.dim_3, analytic_dF.dim_4);

            ////Numerical gradient of loss with respect to bias
            //for (int i = 0; i < test_net.Conv.B.dim_1; i++) {
            //    loss_up = 0.0;
            //    loss_down = 0.0;

            //    test_net.Conv.B.values[i] += h;
            //    loss_up = test_net.forward().values[0];
            //    test_net.Conv.B.values[i] -= 2 * h;
            //    loss_down = test_net.forward().values[0];
            //    test_net.Conv.B.values[i] += h;

            //    numeric_dB.values[i] = (loss_up - loss_down) / (2 * h);
            //}
            //// Console.WriteLine(analytic_dB);
            //// Console.WriteLine(numeric_dB);
            //// Console.WriteLine(analytic_dB.difference(numeric_dB));

            //// Numerical gradient of loss with respect to filters
            //for (int i = 0; i < test_net.Conv.F.dim_1; i++) {
            //    for (int j = 0; j < test_net.Conv.F.dim_2; j++) {
            //        for (int k = 0; k < test_net.Conv.F.dim_3; k++) {
            //            for (int l = 0; l < test_net.Conv.F.dim_4; l++) {
            //                loss_up = 0.0;
            //                loss_down = 0.0;

            //                test_net.Conv.F.values[test_net.Conv.F.index(i, j, k, l)] += h;
            //                loss_up = test_net.forward().values[0];
            //                test_net.Conv.F.values[test_net.Conv.F.index(i, j, k, l)] -= 2 * h;
            //                loss_down = test_net.forward().values[0];
            //                test_net.Conv.F.values[test_net.Conv.F.index(i, j, k, l)] += h;

            //                numeric_dF.values[numeric_dF.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h);
            //            }
            //        }
            //    }
            //}
            //// Console.WriteLine(analytic_dF);
            //// Console.WriteLine(numeric_dF);
            //// Console.WriteLine(analytic_dF.difference(numeric_dF));

            //// Numerical gradient of loss with respect to input
            //for (int i = 0; i < test_net.I.dim_1; i++) {
            //    for (int j = 0; j < test_net.I.dim_2; j++) {
            //        for (int k = 0; k < test_net.I.dim_3; k++) {
            //            for (int l = 0; l < test_net.I.dim_4; l++) {
            //                loss_up = 0.0;
            //                loss_down = 0.0;

            //                test_net.I.values[test_net.I.index(i, j, k, l)] += h;
            //                loss_up = test_net.forward().values[0];
            //                test_net.I.values[test_net.I.index(i, j, k, l)] -= 2 * h;
            //                loss_down = test_net.forward().values[0];
            //                test_net.I.values[test_net.I.index(i, j, k, l)] += h;

            //                numeric_dI.values[numeric_dI.index(i, j, k, l)] = (loss_up - loss_down) / (2 * h);
            //            }
            //        }
            //    }
            //}
            ////Console.WriteLine(analytic_dI);
            ////Console.WriteLine(numeric_dI);
            //// Console.WriteLine(analytic_dI.difference(numeric_dI));
        }
    }
}