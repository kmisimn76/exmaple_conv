#include "kernels/spie_header_cl.h"

#define M_GROUP 4

// conv k1s1 111
__kernel void cl_conv1x1_kernel(
		__read_only  image2d_t	src_tensor,
		__write_only image2d_t	dst_tensor,
#ifndef USE_IMAGE_WEIGHT
		__global 	 float4*	weights_buffer,
#else
		__read_only	 image2d_t	weights_buffer,
#endif
		__global 	 float*		biases_buffer,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,		// K,    stride, M,   C
					 int 		relu ,
					 float		nnzg_threshold
#ifdef MAKE_LUT_IN_KERNEL
					 ,
					 INVALID
		__global     ushort*    if_idx,
		__global     int*    nnzg_per_hw
#endif
					 ) 
{
	int Y = get_global_id(0) * 2;
	int X = get_global_id(1) * 1;
  	int Z = get_global_id(2) * 2;

	if (X >= (dst_tensor_shape.x >> 2) || Y >= dst_tensor_shape.y || Z >= (dst_tensor_shape.z<<2)) {
		return;
	}

#ifdef MAKE_LUT_IN_KERNEL
	 __global ushort* nzg_val = if_idx + (X * dst_tensor_shape.y + Y) * src_tensor_shape.z;
	 float iszero;
#endif

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r010 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3

	ACCUM_FLT4 r001 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r011 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	__global float4* weights_cache;
#ifndef USE_IMAGE_WEIGHT
	__global float4* filters_loc0 = weights_buffer + Z * 4 * src_tensor_shape.w;
	__global float4* filters_loc1 = weights_buffer + (Z+1) * 4 * src_tensor_shape.w;
#endif

	int xc0 = X * 1;
	int yc0 = Y * 1;

	int s = 0;
	int c = 0;
	do {
		float4 src0_00;
		float4 src0_01;
		float4 src1_00;
		float4 src1_01;
		float4 src2_00;
		float4 src2_01;
		float4 src3_00;
		float4 src3_01;

    	src0_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s));
    	src0_01 = read_imagef(src_tensor, smp_zero, (int2)((yc0+1), (xc0) * kernel_args.w + s));
    	src1_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+1));
    	src1_01 = read_imagef(src_tensor, smp_zero, (int2)((yc0+1), (xc0) * kernel_args.w + s+1));
    	src2_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+2));
    	src2_01 = read_imagef(src_tensor, smp_zero, (int2)((yc0+1), (xc0) * kernel_args.w + s+2));
    	src3_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+3));
    	src3_01 = read_imagef(src_tensor, smp_zero, (int2)((yc0+1), (xc0) * kernel_args.w + s+3));
#ifndef USE_IMAGE_WEIGHT
		weights_cache = filters_loc0;
		float4 f0 = weights_cache[0];
		weights_cache = filters_loc1;
		float4 f1 = weights_cache[0];
		filters_loc0 += 1;
		filters_loc1 += 1;
#else
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z));
		float4 f1 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z+1));
		c++;
#endif

		r000 += f0.x * src0_00;
		r000 += f0.y * src1_00;
		r000 += f0.z * src2_00;
		r000 += f0.w * src3_00;
		r010 += f0.x * src0_01;
		r010 += f0.y * src1_01;
		r010 += f0.z * src2_01;
		r010 += f0.w * src3_01;
		r001 += f1.x * src0_00;
		r001 += f1.y * src1_00;
		r001 += f1.z * src2_00;
		r001 += f1.w * src3_00;
		r011 += f1.x * src0_01;
		r011 += f1.y * src1_01;
		r011 += f1.z * src2_01;
		r011 += f1.w * src3_01;

		s += 4;
	} while (s < kernel_args.w);
//if(X==0&&Y==0&&Z==0)
//	printf("%d,%d,%d %d %lf %d\n",dst_tensor_shape.x ,dst_tensor_shape.y,dst_tensor_shape.z, kernel_args.z, r000.x, kernel_args.w);

	{
		FLT4 res = TO_FLT4(r000) + biases_buffer[Z];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z)), res);
#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+0)], 1);
			//int idx = nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+0)]++;
			nzg_val[idx] = Z;
#else
			nzg_val[Z] = 1;
#endif
		}
#endif
	}
	if(Y+1 < dst_tensor_shape.y) {
		FLT4 res = TO_FLT4(r010) + biases_buffer[Z];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)((Y+1), X * kernel_args.z + (Z)), res);
#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+1)], 1);
			//int idx = nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+1)]++;
			nzg_val[idx] = Z;
#else
			nzg_val[Z] = 1;
#endif
		}
#endif
	}

	if(Z+1 >= (dst_tensor_shape.z<<2)) return;
	{
		FLT4 res = TO_FLT4(r001) + biases_buffer[Z+1];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z+1)), res);
#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+0)], 1);
			//int idx = nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+0)]++;
			nzg_val[idx] = Z+1;
#else
			nzg_val[Z] = 1;
#endif
		}
#endif
	}
	if(Y+1 < dst_tensor_shape.y) {
		FLT4 res = TO_FLT4(r011) + biases_buffer[Z+1];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)((Y+1), X * kernel_args.z + (Z+1)), res);
#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+1)], 1);
			//int idx = nnzg_per_hw[(X+0) * dst_tensor_shape.y + (Y+1)]++;
			nzg_val[idx] = Z+1;
#else
			nzg_val[Z] = 1;
#endif
		}
#endif
	}
}
