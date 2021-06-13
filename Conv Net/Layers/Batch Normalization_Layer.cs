using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Batch_Normalization_Layer : Base_Layer {

        public override bool trainable_parameters { get; }
        public override bool test_train_mode { get; }

        private int N; // effective batch size
        private int D; // elements in a single sample
        private int I_sample, I_rows, I_columns, I_channels;
        private bool is_conv;
        private Double momentum;
        
        private Tensor I_hat;
        private Tensor mean;
        private Tensor variance;
        private Tensor running_mean;
        private Tensor running_variance;
        public Tensor inverse_stdev; // 1 / Sqrt(variance + epsilon)
        public override Tensor B { get; set; } // beta
        public override Tensor W { get; set; } // gamma
        public override Tensor dB { get; set; }
        public override Tensor dW { get; set; }
        public override Tensor V_dB { get; set; }
        public override Tensor V_dW { get; set; }
        public override Tensor S_dB { get; set; }
        public override Tensor S_dW { get; set; }

        private Double EPSILON = 0.00001;
        public Batch_Normalization_Layer(int element) {

            this.trainable_parameters = true;
            this.test_train_mode = true;

            this.B = new Tensor(2, 1, element);
            this.dB = new Tensor(2, 1, element);
            this.V_dB = new Tensor(2, 1, element);
            this.S_dB = new Tensor(2, 1, element);

            this.W = new Tensor(2, 1, element);
            this.dW = new Tensor(2, 1, element);
            this.V_dW = new Tensor(2, 1, element);
            this.S_dW = new Tensor(2, 1, element);

            this.running_mean = new Tensor(2, 1, element);
            this.running_variance = new Tensor(2, 1, element);

            this.momentum = 0.9;

            // gamma initialized to 1, beta initialized to 0
            for (int i = 0; i < this.W.values.Length; i++) {
                this.W.values[i] = 1;
            }
            for (int i=0; i < this.B.values.Length; i++) {
                this.B.values[i] = 0;
            }
        }

        public override Tensor forward(Tensor I, bool is_training) {

            /// For batch norm after a fully connected layer, input Tensor will have dimensions [sample x layer size]
            /// For each element in (D = layer size) calculate its mean across (N = sample)

            /// For batch norm after a convolution layer, input Tensor will have dimensions [sample x row x column x channel], reshape to [(sample * row * column) x channel]
            /// For each element in (D = channel) calculate its mean across (N = sample * row * column)
            if (I.dimensions == 4) {
                this.is_conv = true;
                this.I_sample = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;

                I.dimensions = 2;
                I.dim_1 = this.I_sample * this.I_rows * this.I_columns;
                I.dim_2 = this.I_channels;
                I.dim_3 = 1;
                I.dim_4 = 1;
            }
            this.N = I.dim_1;
            this.D = I.dim_2;

            if (is_training == true) {
                
                this.I_hat = new Tensor(2, this.N, this.D);
                this.mean = new Tensor(2, 1, this.D);
                this.variance = new Tensor(2, 1, this.D);
                this.inverse_stdev = new Tensor(2, 1, this.D);

                // Calculate mean
                for (int i = 0; i < this.N; i++) {
                    for (int j = 0; j < this.D; j++) {
                        this.mean.values[j] += (I.values[i * this.D + j] / this.N);
                    }
                }
                
                // Calculate variance 
                for (int i = 0; i < this.N; i++) {
                    for (int j = 0; j < this.D; j++) {
                        this.variance.values[j] += (Math.Pow((I.values[i * this.D + j] - this.mean.values[j]), 2) / this.N);
                    }
                }

                // Update running mean and variance
                for (int i=0; i < this.D; i ++) {
                    this.running_mean.values[i] = (this.momentum * this.running_mean.values[i] + (1 - this.momentum) * this.mean.values[i]);
                    this.running_variance.values[i] = (this.momentum * this.running_variance.values[i] + (1 - this.momentum) * this.variance.values[i]);
                }

                // Calculate inverse standard deviation
                for (int i=0; i < this.D; i++) {
                    this.inverse_stdev.values[i] = 1 / Math.Sqrt(this.variance.values[i] + this.EPSILON);
                    
                }

                // I_hat = (z - mean)/(sqrt(variance) + epsilon) 
                for (int i = 0; i < this.N; i++) {
                    for (int j = 0; j < this.D; j++) {
                        this.I_hat.values[i * this.D + j] = (I.values[i * this.D + j] - this.mean.values[j]) * this.inverse_stdev.values[j];
                    }
                }

                // Output = gamma * ((z - mean)/(sqrt(variance) + epsilon))+ beta
                for (int i = 0; i < this.N; i++) {
                    for (int j = 0; j < this.D; j++) {
                        I.values[i * this.D + j] = this.W.values[j] * (this.I_hat.values[i * this.D + j]) + this.B.values[j];
                    }
                }
                
            } else {
                
                for (int i=0; i < this.N; i++) {
                    for (int j=0; j < this.D; j++) {
                        I.values[i * this.D + j] = this.W.values[j] * ((I.values[i * this.D + j] - this.running_mean.values[j]) / Math.Sqrt(this.running_variance.values[j] + this.EPSILON) + this.B.values[j]);
                    }
                }
            }
            // Reshape output if convolutional layer
            if (is_conv == true) {
                I.dimensions = 4;
                I.dim_1 = I_sample; I.dim_2 = I_rows; I.dim_3 = I_columns; I.dim_4 = I_channels;
            }

            return I;
        }


        public override Tensor backward (Tensor dO) {

 
            if (this.is_conv == true) {
                dO.dimensions = 2;
                dO.dim_1 = I_sample * I_rows * I_columns;
                dO.dim_2 = I_channels;
                dO.dim_3 = 1;
                dO.dim_4 = 1;
            }

            Tensor column_vector_1 = Utils.column_vector_1(this.N);
            Tensor row_vector_1 = Utils.row_vector_1(this.N);
            

            // dB [1 x D] = 1_row [1 x N] * dO [N x D]
            
            this.dB = Utils.dgemm_cs(row_vector_1, dO, dB);

            // dW [1 x D] = 1_row [1 x N] * (dO [N x D] * I_hat [N x D])
            this.dW = Utils.dgemm_cs(row_vector_1, Utils.elementwise_product(dO, this.I_hat), this.dW);

            // dI [N x D] = 1 / N * (1_column [N x 1] * (W [1 x D] / Sqrt(variance [1 x D] + epsilon))) * ((N * dO [N x D]) - (column_1 [N x 1] * dB [1 x D]) - ((column_1 [N x 1] * dW [1 x D]) * I_hat [N x D]))
            Tensor dI = new Tensor(2, this.N, this.D);

            Tensor left_side = new Tensor(2, this.N, this.D);        
            left_side = Utils.scalar_product(1.0/this.N, Utils.dgemm_cs(column_vector_1, Utils.elementwise_product(this.W, this.inverse_stdev), left_side));

            Tensor part1 = Utils.scalar_product(this.N, dO);
            Tensor part2 = new Tensor(2, this.N, this.D);
            part2 = Utils.dgemm_cs(column_vector_1, this.dB, part2);
            Tensor part3 = new Tensor(2, this.N, this.D);
            part3 = Utils.elementwise_product(Utils.dgemm_cs(column_vector_1, this.dW, part3), this.I_hat);
            Tensor right_side = Utils.elementwise_subtract(Utils.elementwise_subtract(part1, part2), part3);

            dI = Utils.elementwise_product(left_side, right_side);

            if (is_conv == true) {
                dI.dimensions = 4;
                dI.dim_1 = I_sample; dI.dim_2 = I_rows; dI.dim_3 = I_columns; dI.dim_4 = I_channels;
            }
            return dI;
        }
    }

}
