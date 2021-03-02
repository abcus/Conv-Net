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
    }


}
