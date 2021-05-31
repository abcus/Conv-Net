using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Batch_Normalization_Layer {

        private int effective_sample_size;
        private int element;
        private int I_sample, I_rows, I_columns, I_channels;
        private bool is_training;
        private bool is_conv;

        private Tensor mean;
        private Tensor variance;
        private Tensor gamma;
        private Tensor beta;

        private Double EPSILON = 0.00000001;
        public Batch_Normalization_Layer() {

        }

        public Tensor forward(Tensor I) {

            is_training = true;

            if (is_training == true) {
                /// For batch norm after a fully connected layer, input Tensor will have dimensions [sample, layer size]
                /// For each element in (layer size) calculate its mean across effective_sample_size = sample

                /// For batch norm after a convolution layer, input Tensor will have dimensions [sample, row, column, channel], reshape to [sample * row * column, channel]
                /// For each element in (channel) calculate its mean across effective_sample_size = (sample x row x column)
                if (I.dimensions == 4) {
                    this.is_conv = true;
                    I_sample = I.dim_1; I_rows = I.dim_2; I_columns = I.dim_3; I_channels = I.dim_4;
                    
                    I.dimensions = 2;
                    I.dim_1 = I_sample * I_rows * I_columns;
                    I.dim_2 = I_channels;
                    I.dim_3 = 1;
                    I.dim_4 = 1;
                }
                effective_sample_size = I.dim_1;
                element = I.dim_2;

                mean = new Tensor(1, element);
                variance = new Tensor(1, element);
                gamma = new Tensor(1, element);
                beta = new Tensor(1, element);
                for (int i = 0; i < gamma.values.Count(); i++) {
                    gamma.values[i] = 1.0;
                }


                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        mean.values[j] += (I.values[i * element + j] / effective_sample_size);
                    }
                }

                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        variance.values[j] += (Math.Pow((I.values[i * element + j] - mean.values[j]), 2) / effective_sample_size);
                    }
                }

                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        I.values[i * element + j] = this.gamma.values[j] * ((I.values[i * element + j] - this.mean.values[j]) / Math.Sqrt(this.variance.values[j] + this.EPSILON)) + this.beta.values[j];
                    }
                }
                if (is_conv == true) {
                    I.dimensions = 4;
                    I.dim_1 = I_sample; I.dim_2 = I_rows; I.dim_3 = I_columns; I.dim_4 = I_channels;
                }
                return I;
            }
            return null;
        }


        public Tensor backward (Tensor dO) {
            return null;
        }
    }

}
