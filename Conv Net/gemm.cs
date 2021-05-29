using System;
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
            for (int i = 0; i < T.dim_1 * T.dim_2 * T.dim_3 * T.dim_4; i++) {
                this.T.values[i] = i / 20.0;
            }

            int image_samples = 2;
            int filter_nums = 2;
            int filter_rows = 3;
            int filter_columns = 3;
            int filter_channels = 2;
            int padding = 4;
            int dilation = 3;
            int stride = 2;

            this.Conv = new Convolution_Layer(filter_channels, filter_nums, filter_rows, filter_columns, true, padding, stride, dilation);
            this.MSE = new Mean_Squared_Loss();

            for (int i = 0; i < this.Conv.F_num * this.Conv.F_rows * this.Conv.F_columns * this.Conv.F_channels; i++) {
                this.Conv.F.values[i] = i / 100.0;
            }

            Tensor FF = Conv.F.F_2_col();
            // Console.WriteLine(FF);
            I = I.pad(padding);
            //Console.WriteLine(I);
            I = I.im_2_col(filter_rows, filter_columns, filter_channels, dilation, stride, image_samples);
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
