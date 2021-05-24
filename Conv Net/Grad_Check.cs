using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    static class Grad_Check {

        public static void forward () {
            Tensor I = new Tensor(4, 1, 6, 6, 2);
            for (int i = 0; i < 6; i++) {
                for (int j=0; j < 6; j++) {
                    for (int k=0; k < 2; k++) {
                        I.values[i * 6 * 2 + j * 2 + k] = ((i + j + 1) * (k+1) / 10.0);
                    }
                }
            }

            Tensor F = new Tensor(4, 2, 3, 3, 2);
            for (int i=0; i < 2; i++) {
                for (int j=0; j < 3; j++) {
                    for (int k=0; k < 3; k++) {
                        for (int l=0; l < 2; l++) {
                            F.values[i * 3 * 3 * 2 + j * 3 * 2 + k * 2 + l] = (j + k + 1) * (i + 1) * (l + 1) / 10.0;
                        }
                    }
                }
            }

            Tensor B = new Tensor(1, 2);
            B.values[0] = 0.77;
            B.values[1] = 0.85;

            Tensor E = new Tensor(4, 1, 2, 2, 2);
            E.values[0] = 4.7;
            E.values[1] = 8.4;
            E.values[2] = 3.8;
            E.values[3] = 7.2;
            E.values[4] = 5.2;
            E.values[5] = 9.3;
            E.values[6] = 3.1;
            E.values[7] = 5.4;

            Convolution_Layer Conv = new Convolution_Layer(2, 2, 3, 3, true, 1, 3, 2);
            Conv.F = F;
            Conv.B = B;
            Mean_Squared_Loss MSE = new Mean_Squared_Loss();

            Tensor T;
            T = Conv.forward(I);
            T = MSE.forward(T, E);

            Tensor G;
            G = MSE.backward();
            Console.WriteLine(G);

        }
    }
}
