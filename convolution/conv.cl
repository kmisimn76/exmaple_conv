
void kernel convolution(
					global const float* input,
					global const float* weight,
					global const float* bias,
					global 		 float* output,
					int WH_in,
					int WH,
					int RS,
					int C,
					int K)
{
	int h = get_global_id(1);
	int w = get_global_id(2);
	int k = get_global_id(0);
    int idx_output = k*WH*WH + h*WH + w;
	output[idx_output] = bias[k];

	for (int r=0;r<RS;r++) {
		for (int s=0;s<RS;s++) {
			for (int c=0;c<C;c++) {
				int idx_input = c*WH_in*WH_in + (h+r)*WH_in + (w+s);
				int idx_weight = k*C*RS*RS + c*RS*RS + r*RS + s;
				output[idx_output] += input[idx_input] * weight[idx_weight];
			}
		}
	}
}
