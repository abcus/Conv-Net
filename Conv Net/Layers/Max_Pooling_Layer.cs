using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Max_Pooling_Layer {

        int numFilterRows;
        int numFilterColumns;
        int stride;
        Double[,,] input;

        Tensor input_tensor;

        public Max_Pooling_Layer (int numFilterRows = 2, int numFilterColumns = 2, int stride = 2) {
            this.numFilterRows = numFilterRows;
            this.numFilterColumns = numFilterColumns;
            this.stride = stride;
        }
        
        public Double[,,] forward(Double[,,] input) {
            this.input = input;
            int numInputRows = input.GetLength(0);
            int numInputColumns = input.GetLength(1);
            int numInputChannels = input.GetLength(2);

            int numOutputRows = ((numInputRows - this.numFilterRows) / this.stride) + 1;
            int numOutputColumns = ((numInputColumns - this.numFilterColumns) / this.stride + 1);
            int numOutputChannels = numInputChannels;

            Double[,,] output = new Double[numOutputRows, numOutputColumns, numOutputChannels];
            Double max = Double.MinValue;

            for (int i=0; i < numOutputRows; i ++) {
                for (int j=0; j < numOutputColumns; j++) {
                    for (int k=0; k < numOutputChannels; k++) {

                        for (int l = 0; l < numFilterRows; l++) {
                            for (int m=0; m < numFilterColumns; m++) {
                                if (input[i * stride + l, j * stride + m, k] > max) {
                                    max = input[i * stride + l, j * stride + m, k];
                                }
                            }
                        }
                        output[i, j, k] = max;
                        max = Double.MinValue;

                    }
                }
            }
            return output;
        }

        public Tensor forward_tensor(Tensor input) {
            this.input_tensor = input;
            int numInputSamples = input.dim_1;
            int numInputRows = input.dim_2;
            int numInputColumns = input.dim_3;
            int numInputChannels = input.dim_4;

            int numOutputRows = ((numInputRows - this.numFilterRows) / this.stride) + 1;
            int numOutputColumns = ((numInputColumns - this.numFilterColumns) / this.stride + 1);
            int numOutputChannels = numInputChannels;

            Tensor output = new Tensor(4, numInputSamples, numOutputRows, numOutputColumns, numOutputChannels);
            Double max = Double.MinValue;
            for (int sample=0; sample < numInputSamples; sample++) {
                for (int i = 0; i < numOutputRows; i++) {
                    for (int j = 0; j < numOutputColumns; j++) {
                        for (int k = 0; k < numOutputChannels; k++) {

                            for (int l = 0; l < numFilterRows; l++) {
                                for (int m = 0; m < numFilterColumns; m++) {
                                    if (input_tensor.values[sample * (input.dim_2 * input.dim_3 * input.dim_4) + (i * stride + l) * (input.dim_3 * input.dim_4) + (j * stride + m) * (input.dim_4) + k] > max) {
                                        max = input_tensor.values[sample * (input.dim_2 * input.dim_3 * input.dim_4) + (i * stride + l) * (input.dim_3 * input.dim_4) + (j * stride + m) * (input.dim_4) + k];
                                    }
                                }
                            }
                            output.values[sample * (output.dim_2 * output.dim_3 * output.dim_4) + i * (output.dim_3 * output.dim_4) + j * (output.dim_4) + k] = max;
                            max = Double.MinValue;

                        }
                    }
                }
            }
            return output;
        }


        public Double[,,] backward(Double[,,] gradientOutput) {
            int numInputRows = this.input.GetLength(0);
            int numInputColumns = this.input.GetLength(0);
            int numInputChannels = this.input.GetLength(2);
            Double max = Double.MinValue;
            int maxRow = -1;
            int maxColumn = -1;

            // dL/dI
            Double[,,] gradientInput = new Double[numInputRows, numInputColumns, numInputChannels];

            for (int i=0; i <= numInputRows - this.numFilterRows; i+= this.stride) {
                for (int j=0; j <= numInputColumns - this.numFilterColumns; j += this.stride) {
                    for (int k=0; k < numInputChannels; k++) {

                        for (int l = 0; l < numFilterRows; l++) {
                            for (int m = 0; m < numFilterColumns; m++) {
                                if (input[i + l, j + m, k] > max) {
                                    max = input[i + l, j + m, k];
                                    maxRow = i + l;
                                    maxColumn = j + m;
                                }
                            }
                        }
                        gradientInput[maxRow, maxColumn, k] = gradientOutput[i / this.stride, j / this.stride, k];

                        max = Double.MinValue;
                        maxRow = -1;
                        maxColumn = -1;
                    }
                }
            }
            return gradientInput;
        }
        public Tensor backward_tensor(Tensor gradient_output) {
            int input_sample = this.input_tensor.dim_1;
            int input_row = this.input_tensor.dim_2;
            int input_column = this.input_tensor.dim_3;
            int input_channel = this.input_tensor.dim_4;
            Double max = Double.MinValue;
            int maxRow = -1;
            int maxColumn = -1;
            Tensor gradient_input = new Tensor(4, input_sample, input_row, input_column, input_channel);

            for (int sample = 0; sample < input_sample; sample ++ ) {
                for (int i = 0; i <= input_row - this.numFilterRows; i += this.stride) {
                    for (int j = 0; j <= input_column - this.numFilterColumns; j += this.stride) {
                        for (int k = 0; k < input_channel; k++) {

                            for (int l = 0; l < numFilterRows; l++) {
                                for (int m = 0; m < numFilterColumns; m++) {
                                    if (input_tensor.values[sample * (input_tensor.dim_2 * input_tensor.dim_3 * input_tensor.dim_4) + (i + l) * (input_tensor.dim_3 * input_tensor.dim_4) + (j + m) * (input_tensor.dim_4) + k] > max) {
                                        max = input_tensor.values[sample * (input_tensor.dim_2 * input_tensor.dim_3 * input_tensor.dim_4) + (i + l) * (input_tensor.dim_3 * input_tensor.dim_4) + (j + m) * (input_tensor.dim_4) + k];
                                        maxRow = i + l;
                                        maxColumn = j + m;
                                    }
                                }
                            }
                            gradient_input.values[sample * (gradient_input.dim_2 * gradient_input.dim_3 * gradient_input.dim_4) + maxRow * (gradient_input.dim_3 * gradient_input.dim_4) + maxColumn * (gradient_input.dim_4) + k] = gradient_output.values[sample * (gradient_output.dim_2 * gradient_output.dim_3 * gradient_output.dim_4) + (i / this.stride) * (gradient_output.dim_3 * gradient_output.dim_4) + (j / this.stride) * (gradient_output.dim_4) + k];

                            max = Double.MinValue;
                            maxRow = -1;
                            maxColumn = -1;
                        }
                    }
                }
            }
            return gradient_input;
        }
    }


}
