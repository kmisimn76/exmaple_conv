#include "kernels/spie_header_cl.h"

// conv k1s1 111
// k1: kernel size (1x1)
// s1: stride = 1
// 111: per work size (# output values that a work-item produces; grouping is considered; 1 group = 1 value)
__kernel void cl_conv1x1_kernel(
		__read_only  image2d_t	src_tensor,
		__write_only image2d_t	dst_tensor,
#ifndef USE_IMAGE_WEIGHT
		__global 	 float4*	weights_buffer,
#else
		__read_only	 image2d_t	weights_buffer,
#endif
		__global 	 float4*	biases_buffer,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,
					 int		relu) 
{
	int X = get_global_id(0) * 2;	// E-axis (=W)
	int Y = get_global_id(1) * 2;	// F-axis (=H)
  	int Z = get_global_id(2) * 2;	// M-axis (M/4)

	if (X >= dst_tensor_shape.x || Y >= dst_tensor_shape.y || Z >= dst_tensor_shape.z) {
		return;
	}

		// w h c
	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r100 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r010 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r110 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r001 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r101 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r011 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r111 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	__global float4* weights_cache;	// 4 filter values for a specific channel in a vector
#ifndef USE_IMAGE_WEIGHT
	__global float4* filters_loc = weights_buffer + Z * 4 * src_tensor_shape.w;
#endif

	int xc0 = X * 1;
	int xc1 = X * 1 + 1;
	int yc0 = Y * 1;
	int yc1 = Y * 1 + 1;

	int s = 0;
	int c = 0;
	do {	// reduction loop for entire C
		float4 src00;
		float4 src10;
		float4 src01;
		float4 src11;

    	src00 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + s));
    	src10 = read_imagef(src_tensor, smp_zero, (int2)((xc1), (yc0) * src_tensor_shape.w + s));
    	src01 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc1) * src_tensor_shape.w + s));
    	src11 = read_imagef(src_tensor, smp_zero, (int2)((xc1), (yc1) * src_tensor_shape.w + s));
		s += 1;
#ifndef USE_IMAGE_WEIGHT
		INVALID!
		weights_cache = filters_loc;
		r000 += weights_cache[0] * src00.x;	// src00.x is widen; 4 filter values for a channel are calculated
		r000 += weights_cache[1] * src00.y;	// another channel
		r000 += weights_cache[2] * src00.z;	// another channel
		r000 += weights_cache[3] * src00.w;	// another channel
		filters_loc += 4;
#else
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z));
		float4 f1 = read_imagef(weights_buffer, smp_zero, (int2)(c+1, Z));
		float4 f2 = read_imagef(weights_buffer, smp_zero, (int2)(c+2, Z));
		float4 f3 = read_imagef(weights_buffer, smp_zero, (int2)(c+3, Z));
		float4 f4 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z+1));
		float4 f5 = read_imagef(weights_buffer, smp_zero, (int2)(c+1, Z+1));
		float4 f6 = read_imagef(weights_buffer, smp_zero, (int2)(c+2, Z+1));
		float4 f7 = read_imagef(weights_buffer, smp_zero, (int2)(c+3, Z+1));

		r000 += f0 * src00.x;
		r100 += f0 * src10.x;
		r010 += f0 * src01.x;
		r110 += f0 * src11.x;
		r000 += f1 * src00.y;
		r100 += f1 * src10.y;
		r010 += f1 * src01.y;
		r110 += f1 * src11.y;
		r000 += f2 * src00.z;
		r100 += f2 * src10.z;
		r010 += f2 * src01.z;
		r110 += f2 * src11.z;
		r000 += f3 * src00.w;
		r100 += f3 * src10.w;
		r010 += f3 * src01.w;
		r110 += f3 * src11.w;
		r001 += f4 * src00.x;
		r101 += f4 * src10.x;
		r011 += f4 * src01.x;
		r111 += f4 * src11.x;
		r001 += f5 * src00.y;
		r101 += f5 * src10.y;
		r011 += f5 * src01.y;
		r111 += f5 * src11.y;
		r001 += f6 * src00.z;
		r101 += f6 * src10.z;
		r011 += f6 * src01.z;
		r111 += f6 * src11.z;
		r001 += f7 * src00.w;
		r101 += f7 * src10.w;
		r011 += f7 * src01.w;
		r111 += f7 * src11.w;

		c += 4;
#endif
	} while (s < src_tensor_shape.w);	// loop over C/4

	weights_cache = biases_buffer + Z;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r000) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (Z + 0)), res);
	}
	if (X+1 < dst_tensor_shape.x)
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r100) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 1)), ((Y + 0)) * dst_tensor_shape.z + (Z + 0)), res);
	}
	if (Y+1 < dst_tensor_shape.y)
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r010) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 1)) * dst_tensor_shape.z + (Z + 0)), res);
	}
	if (X+1 < dst_tensor_shape.x && Y+1 < dst_tensor_shape.y)
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r110) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 1)), ((Y + 1)) * dst_tensor_shape.z + (Z + 0)), res);
	}

	if (Z+1 >= dst_tensor_shape.z) return;
	weights_cache = biases_buffer + Z+1;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r001) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (Z + 1)), res);
	}
	if (X+1 < dst_tensor_shape.x)
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r101) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 1)), ((Y + 0)) * dst_tensor_shape.z + (Z + 1)), res);
	}
	if (Y+1 < dst_tensor_shape.y)
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r011) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 1)) * dst_tensor_shape.z + (Z + 1)), res);
	}
	if (X+1 < dst_tensor_shape.x && Y+1 < dst_tensor_shape.y)
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r111) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 1)), ((Y + 1)) * dst_tensor_shape.z + (Z + 1)), res);
	}
}
