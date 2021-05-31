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

        private Tensor I_hat;
        private Tensor mean;
        private Tensor variance;
        public Tensor gamma;
        public Tensor beta;

        public Tensor d_gamma;
        public Tensor d_beta;

        private Double EPSILON = 0.00000001;
        public Batch_Normalization_Layer(int element) {
            gamma = new Tensor(1, element);
            beta = new Tensor(1, element);
            for (int i = 0; i < gamma.values.Count(); i++) {
                gamma.values[i] = i + 1;
            }
            for (int i=0; i < beta.values.Count(); i++) {
                beta.values[i] = i + 2;
            }
        }

        public Tensor forward(Tensor I, bool is_training) {

            this.is_training = is_training;

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

                this.I_hat = new Tensor(2, effective_sample_size, element);
                mean = new Tensor(1, element);
                variance = new Tensor(1, element);

                // Calculate mean
                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        mean.values[j] += (I.values[i * element + j] / effective_sample_size);
                    }
                }

                // Calculate variance
                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        variance.values[j] += (Math.Pow((I.values[i * element + j] - mean.values[j]), 2) / effective_sample_size);
                    }
                }

                // I_hat = (z - mean)/(sqrt(variance) + epsilon) 
                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        this.I_hat.values[i * element + j] = (I.values[i * element + j] - this.mean.values[j]) / Math.Sqrt(this.variance.values[j] + this.EPSILON) ;
                    }
                }

                // Output = gamma * ((z - mean)/(sqrt(variance) + epsilon))+ beta
                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        I.values[i * element + j] = this.gamma.values[j] * ((I.values[i * element + j] - this.mean.values[j]) / Math.Sqrt(this.variance.values[j] + this.EPSILON)) + this.beta.values[j];
                    }
                }
                // Reshape output if convolutional layer
                if (is_conv == true) {
                    I.dimensions = 4;
                    I.dim_1 = I_sample; I.dim_2 = I_rows; I.dim_3 = I_columns; I.dim_4 = I_channels;
                }

                return I;
            }
            return null;
        }


        public Tensor backward (Tensor dO) {
            if (this.is_conv == true) {
                dO.dimensions = 2;
                dO.dim_1 = I_sample * I_rows * I_columns;
                dO.dim_2 = I_channels;
                dO.dim_3 = 1;
                dO.dim_4 = 1;

            }
            // dL/dgamma
            Tensor one_vector = Utils.one_vector_1D(effective_sample_size);
            this.d_gamma = new Tensor(1, this.element);
            this.d_gamma = Utils.dgbvm_cs(one_vector, Utils.elementwise_multiply(dO, this.I_hat), this.d_gamma);

            // dL/dbeta
            this.d_beta = new Tensor(1, this.element);
            this.d_beta = Utils.dgbvm_cs(one_vector, dO, d_beta);   

            
            

            if (is_conv == true) {
                dO.dimensions = 4;
                dO.dim_1 = I_sample; dO.dim_2 = I_rows; dO.dim_3 = I_columns; dO.dim_4 = I_channels;
            }
            return dO;
        }
    }

}
