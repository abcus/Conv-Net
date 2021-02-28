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

        public override string ToString() {
            StringBuilder s = new StringBuilder();
            s.Append("rank: " + this.rank + "\nsamples: " + this.num_samples + "\nrows: " + this.num_rows + "\ncolumns: " + this.num_columns + "\nchannels: " + this.num_channels + "\n");
            return s.ToString();            
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
