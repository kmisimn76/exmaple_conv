#include "kernels/spie_header_cl.h"

// conv k1s1 111
// k1: kernel size (1x1)
// s1: stride = 1
// 111: per work size (# output values that a work-item produces; grouping is considered; 1 group = 1 value)
#undef USE_IMAGE_WEIGHT // force to use constant memory
__kernel void cl_conv1x1_kernel(
		__read_only  image2d_t	src_tensor,
		__write_only image2d_t	dst_tensor,
#ifndef USE_IMAGE_WEIGHT
		__constant 	 float4*	weights_buffer,
#else
		__read_only	 image2d_t	weights_buffer,
#endif
		__constant 	 float4*	biases_buffer,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,
					 int		relu, 
					 float		nnzg_threshold
#ifdef MAKE_LUT_IN_KERNEL
					 ,
					 INVALID
		__global     ushort*    if_idx,
		__global     int*    nnzg_per_hw
#endif
						)

{
	int X = get_global_id(0) * 1;	// E-axis (=W)
	int Y = get_global_id(1) * 1;	// F-axis (=H)
// 	int Z = get_global_id(2) * 1;	// M-axis (M/4)
	int Z = 0;

	if (X >= dst_tensor_shape.x || Y >= dst_tensor_shape.y || Z >= 1) {
		return;
	}

	ACCUM_FLT4 r0 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// ACCUM_FLT4 == float4 (TF Lite notation)
	ACCUM_FLT4 r1 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// ACCUM_FLT4 == float4 (TF Lite notation)
	ACCUM_FLT4 r2 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// ACCUM_FLT4 == float4 (TF Lite notation)
	ACCUM_FLT4 r3 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// ACCUM_FLT4 == float4 (TF Lite notation)
	__constant float4* weights_cache;	// 4 filter values for a specific channel in a vector
#ifndef USE_IMAGE_WEIGHT
	__constant float4* filters_loc = weights_buffer;
#endif

	int xc0 = X * 1;
	int yc0 = Y * 1;

	int s = 0;
	int c = 0;
	for(int i=0;i<4;i++) {
		float4 src00;
    	src00 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + s));	// HCWc4 format
		s += 1;
#ifndef USE_IMAGE_WEIGHT
		weights_cache = filters_loc;
		r0 += weights_cache[0] * src00.x;	// src00.x is widen; 4 filter values for a channel are calculated
		r0 += weights_cache[1] * src00.y;	// another channel
		r0 += weights_cache[2] * src00.z;	// another channel
		r0 += weights_cache[3] * src00.w;	// another channel
		r1 += weights_cache[4] * src00.x;	// src00.x is widen; 4 filter values for a channel are calculated
		r1 += weights_cache[5] * src00.y;	// another channel
		r1 += weights_cache[6] * src00.z;	// another channel
		r1 += weights_cache[7] * src00.w;	// another channel
		r2 += weights_cache[8] * src00.x;	// src00.x is widen; 4 filter values for a channel are calculated
		r2 += weights_cache[9] * src00.y;	// another channel
		r2 += weights_cache[10] * src00.z;	// another channel
		r2 += weights_cache[11] * src00.w;	// another channel
		r3 += weights_cache[12] * src00.x;	// src00.x is widen; 4 filter values for a channel are calculated
		r3 += weights_cache[13] * src00.y;	// another channel
		r3 += weights_cache[14] * src00.z;	// another channel
		r3 += weights_cache[15] * src00.w;	// another channel
		filters_loc += 16;
#else
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, i));
		float4 f1 = read_imagef(weights_buffer, smp_zero, (int2)(c+1, i));
		float4 f2 = read_imagef(weights_buffer, smp_zero, (int2)(c+2, i));
		float4 f3 = read_imagef(weights_buffer, smp_zero, (int2)(c+3, i));
		float4 f4 = read_imagef(weights_buffer, smp_zero, (int2)(c+4, i));
		float4 f5 = read_imagef(weights_buffer, smp_zero, (int2)(c+5, i));
		float4 f6 = read_imagef(weights_buffer, smp_zero, (int2)(c+6, i));
		float4 f7 = read_imagef(weights_buffer, smp_zero, (int2)(c+7, i));
		float4 f8 = read_imagef(weights_buffer, smp_zero, (int2)(c+8, i));
		float4 f9 = read_imagef(weights_buffer, smp_zero, (int2)(c+9, i));
		float4 f10 = read_imagef(weights_buffer, smp_zero, (int2)(c+10, i));
		float4 f11 = read_imagef(weights_buffer, smp_zero, (int2)(c+11, i));
		float4 f12 = read_imagef(weights_buffer, smp_zero, (int2)(c+12, i));
		float4 f13 = read_imagef(weights_buffer, smp_zero, (int2)(c+13, i));
		float4 f14 = read_imagef(weights_buffer, smp_zero, (int2)(c+14, i));
		float4 f15 = read_imagef(weights_buffer, smp_zero, (int2)(c+15, i));
		r0 += f0 * src00.x;
		r0 += f1 * src00.y;
		r0 += f2 * src00.z;
		r0 += f3 * src00.w;
		r1 += f4 * src00.x;
		r1 += f5 * src00.y;
		r1 += f6 * src00.z;
		r1 += f7 * src00.w;
		r2 += f8 * src00.x;
		r2 += f9 * src00.y;
		r2 += f10 * src00.z;
		r2 += f11 * src00.w;
		r3 += f12 * src00.x;
		r3 += f13 * src00.y;
		r3 += f14 * src00.z;
		r3 += f15 * src00.w;
#endif
	}

	weights_cache = biases_buffer;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r0) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (0)), res);
	}
	weights_cache = biases_buffer + 1;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r1) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (1)), res);
	}
	weights_cache = biases_buffer + 2;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r2) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (2)), res);
	}
	weights_cache = biases_buffer + 3;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r3) + bias_val;
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (3)), res);
	}
}
