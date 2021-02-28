using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Conv_Net {
    class Tensor {

        // Ranks are: batch_size x rows x columns x channels
        public int rank;
        public int num_samples;
        public int num_rows;
        public int num_columns;
        public int num_channels;
        public Double[] data;

        public Tensor (int rank, int num_samples, int num_rows, int num_columns, int num_channels) {
            this.rank = rank;
            this.num_samples = num_samples;
            this.num_rows = num_rows;
            this.num_columns = num_columns;
            this.num_channels = num_channels;
            data = new Double[this.num_samples * this.num_rows * this.num_columns * this.num_channels];
        }

        public Double get (int sample, int row, int column, int channel) {
            return data[sample * num_rows * num_columns * num_channels + row * num_columns * num_channels + column * num_channels + channel];
        }

        public void set (int sample, int row, int column, int channel, Double value) {
            data[sample * num_rows * num_columns * num_channels + row * num_columns * num_channels + column * num_channels + channel] = value;
        }

        public Tensor partition(int sample_i, int partition_size) {
            Tensor t = new Tensor(this.rank, partition_size, this.num_rows, this.num_columns, this.num_channels);
            for (int i = 0; i < partition_size; i++) {
                for (int j=0; j < this.num_rows; j++) {
                    for (int k=0; k < this.num_columns; k++) { 
                        for (int l=0; l < this.num_channels; l++) {
                            t.data[i * num_rows * num_columns * num_channels + j * num_columns * num_channels + k * num_channels + l] =
                                this.data[(i + sample_i) * num_rows * num_columns * num_channels + j * num_columns * num_channels + k * num_channels + l];
                        }
                    }
                }
            }
            return t;
        }


        public override string ToString() {
            StringBuilder s = new StringBuilder();
            s.Append("\nsamples: " + this.num_samples + "\nrows: " + this.num_rows + "\ncolumns: " + this.num_columns + "\nchannels: " + this.num_channels + "\n");


            //for (int i=0; i < this.num_samples; i ++) {
            //    s.Append("(");
            //    for (int j = 0; j < this.num_rows; j++) {
            //        s.Append("[");
            //        for (int k = 0; k < this.num_columns; k++) {
            //            s.Append("{");
            //            for (int l = 0; l < this.num_channels; l++) {
            //                s.Append(this.data[i * num_rows * num_columns * num_channels + j * num_columns * num_channels + k * num_channels + l]);
            //                if (l < this.num_channels - 1) {
            //                    s.Append(", ");
            //                } else {
            //                    s.Append("}");
            //                }
            //            }
            //            if (k < this.num_columns - 1) {
            //                s.Append(", ");
            //            } else {
            //                s.Append("");
            //            }
            //        }
            //        s.Append("]");
            //        if (j < this.num_rows - 1) {
            //            s.Append(",\n");
            //        }
            //        s.Append("");
            //    }
            //    s.Append(")\n");
            //}

            return s.ToString();            
        }

        public bool Equals(Tensor t) {
            bool equals = true;
            if (this.rank != t.rank || this.num_samples != t.num_samples || this.num_rows != t.num_rows || this.num_columns != t.num_columns || this.num_channels != t.num_channels) {
                return false;
            }
            for (int i = 0; i < this.num_samples; i++) {
                for (int j = 0; j < this.num_rows; j++) {
                    for (int k = 0; k < this.num_columns; k++) {
                        for (int l = 0; l < this.num_channels; l++) {
                            if (this.data[i * num_rows * num_columns * num_channels + j * num_columns * num_channels + k * num_channels + l] != t.data[i * num_rows * num_columns * num_channels + j * num_columns * num_channels + k * num_channels + l]) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        public Double[][,,] convert_to_array () {
            Double[][,,] array = new Double[this.num_samples][,,];
            for (int i=0; i < num_samples; i++) {
                Double[,,] temp = new Double[this.num_rows, this.num_columns, this.num_channels];
                for (int j=0; j < this.num_rows; j++) {
                    for (int k=0; k < this.num_columns; k++) {
                        for (int l=0; l < this.num_channels; l++) {
                            temp[j, k, l] = data[i * num_rows * num_columns * num_channels + j * num_columns * num_channels + k * num_channels + l];
                        }
                    }
                }
                array[i] = temp;
            }
            return array;
        }
    }
}
