namespace Conv_Net {
    class Flatten_Layer {

        private int I_dimensions, I_samples, I_rows, I_columns, I_channels;

        public Flatten_Layer() {
        }

        public Tensor forward(Tensor I) {
            this.I_dimensions = I.dimensions; this.I_samples = I.dim_1; this.I_rows = I.dim_2; this.I_columns = I.dim_3; this.I_channels = I.dim_4;
            I.dimensions = 2; I.dim_1 = I.dim_1; I.dim_2 = I.dim_2 * I.dim_3 * I.dim_4; I.dim_3 = 1; I.dim_4 = 1;
            return I;
        }

        public Tensor backward(Tensor dO) {
            dO.dimensions = 4; dO.dim_1 = this.I_samples; dO.dim_2 = this.I_rows; dO.dim_3 = this.I_columns; dO.dim_4 = this.I_channels;
            return dO;            
        }
    }
}
