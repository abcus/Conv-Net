using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Batch_Normalization_Layer : Layer {

        private int effective_sample_size;
        private int element;
        private int I_sample, I_rows, I_columns, I_channels;
        private bool is_training;
        private bool is_conv;

        
        private Tensor I_hat;
        private Tensor mean;
        private Tensor variance;
        public Tensor inverse_stdev; // 1 / Sqrt(variance + epsilon)
        public override Tensor B { get; set; } // beta
        public override Tensor W { get; set; } // gamma
        public override Tensor dB { get; set; }
        public override Tensor dW { get; set; }
        

        private Double EPSILON = 0.00001;
        public Batch_Normalization_Layer(int element) {
            W = new Tensor(1, element);
            B = new Tensor(1, element);
            // gamma initialized to 1, beta initialized to 0
            for (int i = 0; i < W.values.Length; i++) {
                W.values[i] = 1;
            }
            for (int i=0; i < B.values.Length; i++) {
                B.values[i] = 0;
            }
        }

        override public Tensor forward(Tensor I, bool is_training) {

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
                inverse_stdev = new Tensor(1, element);

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

                // Calculate inverse standard deviation
                for (int i=0; i < element; i++) {
                    this.inverse_stdev.values[i] = 1 / Math.Sqrt(this.variance.values[i] + this.EPSILON);
                    
                }

                // I_hat = (z - mean)/(sqrt(variance) + epsilon) 
                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        this.I_hat.values[i * element + j] = (I.values[i * element + j] - this.mean.values[j]) * this.inverse_stdev.values[j];
                    }
                }

                // Output = gamma * ((z - mean)/(sqrt(variance) + epsilon))+ beta
                for (int i = 0; i < effective_sample_size; i++) {
                    for (int j = 0; j < element; j++) {
                        I.values[i * element + j] = this.W.values[j] * ((I.values[i * element + j] - this.mean.values[j]) / Math.Sqrt(this.variance.values[j] + this.EPSILON)) + this.B.values[j];
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


        override public Tensor backward (Tensor dO) {

 
            if (this.is_conv == true) {
                dO.dimensions = 2;
                dO.dim_1 = I_sample * I_rows * I_columns;
                dO.dim_2 = I_channels;
                dO.dim_3 = 1;
                dO.dim_4 = 1;
            }

            // dL/dgamma
            Tensor one_vector = Utils.one_vector_1D(effective_sample_size);
            Tensor column_vector_1 = Utils.column_vector_1(effective_sample_size);

            // dL/dbeta
            this.dB = new Tensor(1, this.element);
            this.dB = Utils.dgbvm_cs(one_vector, dO, dB);

            this.dW = new Tensor(1, this.element);
            this.dW = Utils.dgbvm_cs(one_vector, Utils.elementwise_product(dO, this.I_hat), this.dW);
            

            // dL/dI
            Tensor dI = new Tensor(2, this.effective_sample_size, this.element);

            this.dB.dimensions = 2; this.dB.dim_2 = this.dB.dim_1; this.dB.dim_1 = 1;
            this.dW.dimensions = 2; this.dW.dim_2 = this.dW.dim_1; this.dW.dim_1 = 1;
            this.inverse_stdev.dimensions = 2; this.inverse_stdev.dim_2 = this.inverse_stdev.dim_1; this.inverse_stdev.dim_1 = 1;
            this.W.dimensions = 2; this.W.dim_2 = this.W.dim_1; this.W.dim_1 = 1;

            Tensor left_side = new Tensor(2, this.effective_sample_size, this.element);        
            left_side = Utils.scalar_product(1.0/this.effective_sample_size ,Utils.dgemm_cs(column_vector_1, Utils.elementwise_product(this.W, this.inverse_stdev), left_side));

            Tensor part1 = Utils.scalar_product(this.effective_sample_size, dO);
            Tensor part2 = new Tensor(2, this.effective_sample_size, this.element);
            part2 = Utils.dgemm_cs(column_vector_1, this.dB, part2);
            Tensor part3 = new Tensor(2, this.effective_sample_size, this.element);
            part3 = Utils.elementwise_product(Utils.dgemm_cs(column_vector_1, this.dW, part3), this.I_hat);
            Tensor right_side = Utils.subtract(Utils.subtract(part1, part2), part3);

            dI = Utils.elementwise_product(left_side, right_side);

            this.dB.dimensions = 1; this.dB.dim_1 = this.dB.dim_2; this.dB.dim_2 = 1;
            this.dW.dimensions = 1; this.dW.dim_1 = this.dW.dim_2; this.dW.dim_2 = 1;
            this.W.dimensions = 1; this.W.dim_1 = this.W.dim_2; this.W.dim_2 = 1;
            this.inverse_stdev.dimensions = 1; this.inverse_stdev.dim_1 = this.inverse_stdev.dim_2; this.inverse_stdev.dim_2 = 1;

            if (is_conv == true) {
                dI.dimensions = 4;
                dI.dim_1 = I_sample; dI.dim_2 = I_rows; dI.dim_3 = I_columns; dI.dim_4 = I_channels;
            }
            return dI;
        }
    }

}
