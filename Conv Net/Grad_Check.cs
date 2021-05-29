using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Conv_Net {

    class test_CNN {

        public Convolution_Layer Conv; 
        public Mean_Squared_Loss MSE;
        public Tensor I, T;

        public int I_samples, I_rows, I_columns, I_channels;
        public int F_num, F_rows, F_columns, F_channels;
        int pad_size;
        int stride;
        int dilation;
        int O_samples, O_rows, O_columns, O_channels;

        public test_CNN() {

            // Input: 2 samples x 5 rows x 5 columns x 4 channels
            // Filters: 3 num x 3 rows x 3 columns x 4 channels
            
            // Padding: 4
            // Stride 2
            // Dilation: 3

            // Output size: 2 samples x 5 rows x 5 columns x 3 channels

            I_samples = 2; I_rows = 5; I_columns = 5; I_channels = 4;
            F_num = 3; F_rows = 3; F_columns = 3; F_channels = 4;
            
            pad_size = 4;
            stride = 2;
            dilation = 3;

            O_samples = I_samples;
            O_rows = (I_rows + 2 * pad_size - F_rows * dilation + dilation - 1) / stride + 1;
            O_columns = (I_columns + 2 * pad_size - F_columns * dilation + dilation - 1) / stride + 1;
            O_channels = F_num;

            // test input tensor
            this.I = new Tensor(4, I_samples, I_rows, I_columns, I_channels);
            for (int i = 0; i < I_samples; i++) {
                for (int j=0; j < I_rows; j++) {
                    for (int k=0; k < I_columns; k++) {
                        for (int l=0; l < I_channels; l++) {
                            I.values[I.index(i, j, k, l)] = i + l * I.dim_2 * I.dim_3 + j * I.dim_3 + k + 1;
                        }
                    }
                }
            } 

            // test target tensor
            this.T = new Tensor(4, O_samples, O_rows, O_columns, O_channels);
            for (int i = 0; i < O_samples; i++) {
                for (int j = 0; j < O_rows; j++) {
                    for (int k = 0; k < O_columns; k++) {
                        for (int l = 0; l < O_channels; l++) {
                            T.values[T.index(i, j, k, l)] = i * T.dim_2 * T.dim_3 * T.dim_4 + l * T.dim_2 * T.dim_3 + j * T.dim_3 + k + 1;
                        }
                    }
                }
            }

            this.Conv = new Convolution_Layer(I_channels, F_num, F_rows, F_columns, true, pad_size, stride, dilation);
            this.MSE = new Mean_Squared_Loss();

            // Set filters
            // test target tensor
            for (int i = 0; i < F_num; i++) {
                for (int j = 0; j < F_rows; j++) {
                    for (int k = 0; k < F_columns; k++) {
                        for (int l = 0; l < F_channels; l++) {
                            this.Conv.F.values[this.Conv.F.index(i, j, k, l)] = i * F_rows * F_columns * F_channels + l * F_rows * F_columns + j * F_columns + k + 1;
                        }
                    }
                }
            }


            // Set biases
            for (int i=0; i < F_num; i++) {
                this.Conv.B.values[i] = (i + 1);
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
            
            //Tensor dO_2d = Utils.dO_to_matrix(Z);
                        
            // stride of filter dO (2nd last parameter) is equal to dilation of F
            // dilation of filter dO (last parameter) is equal to stride of F
            //Tensor I_2d = Utils.I_to_matrix_backprop(Conv.I, Z.dim_2, Z.dim_3, F_rows, F_columns, F_channels, dilation, stride);        
            //Tensor dF_2d = new Tensor(2, F_num, F_rows * F_columns * F_channels);
            //dF_2d = Utils.dgemm_cs(dO_2d, I_2d, dF_2d);
            //Tensor dF = Utils.dF_matrix_to_tensor(dF_2d, F_num, F_rows, F_columns, F_channels);
            //Console.WriteLine(dF);

            //Tensor F_rotated_2d = Utils.F_rotated_2_col(Conv.F.rotate_180());
            //Tensor dO_padded_2d = Utils.dO_dilated_padded_to_matrix(Z.dilate(stride).pad(F_rows * dilation - dilation).unpad(pad_size), F_num, F_rows, F_columns, I_samples, I_rows, I_columns, dilation);
            //Tensor dI_2d = new Tensor(2, I_channels, I_samples * I_rows * I_columns);
            //dI_2d = Utils.dgemm_cs(F_rotated_2d, dO_padded_2d, dI_2d);
            //Tensor dI = Utils.matrix_to_tensor(dI_2d, I_samples, I_rows, I_columns, I_channels);
            //Console.WriteLine(dI);

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
            // Console.WriteLine(analytic_dB.difference(numeric_dB));

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
            //Console.WriteLine("--------------------------------------");
            //Console.WriteLine("ANALYTIC DF");
            //Console.WriteLine(analytic_dF);
            // Console.WriteLine(numeric_dF);
            // Console.WriteLine(analytic_dF.difference(numeric_dF));

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
            //Console.WriteLine("--------------------------------------");
            //Console.WriteLine("ANALYTIC DI");
            //Console.WriteLine(analytic_dI);
            // Console.WriteLine(numeric_dI);
            // Console.WriteLine(analytic_dI.difference(numeric_dI));
        }
    }
}
