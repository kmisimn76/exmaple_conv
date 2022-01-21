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
//		__global 	 float4*	biases_buffer,
		__global 	 float*		biases_buffer,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,		// K,    stride, M,   C
					 int 		relu,
					 float		nnzg_threshold
#ifdef MAKE_LUT_IN_KERNEL
					 ,
		__global     ushort*    if_idx,
		__global     int*    nnzg_per_hw
#endif
					 )
{
	int Y = get_global_id(0) * 1;
	int X = get_global_id(1) * 1;
  	int Z = get_global_id(2) * 1;

	if (X >= (dst_tensor_shape.x >> 2) || Y >= dst_tensor_shape.y || Z >= dst_tensor_shape.z) {
		return;
	}

#ifdef MAKE_LUT_IN_KERNEL
	 __global ushort* nzg_val = if_idx + (X * dst_tensor_shape.y + Y) * src_tensor_shape.z;
	 float iszero;
#endif

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r100 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r200 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r300 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	__global float4* weights_cache;
#ifndef USE_IMAGE_WEIGHT
	__global float4* filters_loc = weights_buffer + Z * M_GROUP * dst_tensor_shape.w;
#endif

	int xc0 = X * 1;
	int yc0 = Y * 1;

	int s = 0;
	int c = 0;
	do {
		float4 src00;

    	src00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s));

#ifndef USE_IMAGE_WEIGHT
		weights_cache = filters_loc;
		float4 f0 = weights_cache[0];
		filters_loc += 1;
#else
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c,Z));
		c++;
		//INVALID
#endif
		r000 += f0.x * src00;	// m=0 / c=0 / x=0~3
		r100 += f0.y * src00; // m=1 / c=0 / x=0~3
		r200 += f0.z * src00;
		r300 += f0.w * src00;
    	
		s += 1;
	} while (s < kernel_args.w);

//	weights_cache = biases_buffer + Z;
//	if(Z + 0 >= tensor_shape.z) return;	// no need for this kernel 111
	{
		FLT4 res = TO_FLT4(r000) + biases_buffer[Z * M_GROUP + 0];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		res = select((float4)(0.f), res, nnzg_threshold < fabs(res));
		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 0)), res);

#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[X * dst_tensor_shape.y + Y], 1);
//			int idx = nnzg_per_hw[X * dst_tensor_shape.y + Y]++;
			nzg_val[idx] = (Z * M_GROUP + 0);
#else
			nzg_val[(Z * M_GROUP + 0)] = 1;
#endif
		}
#endif


	}
	{
		FLT4 res = TO_FLT4(r100) + biases_buffer[Z * M_GROUP + 1];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		res = select((float4)(0.f), res, nnzg_threshold < fabs(res));
		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 1)), res);

#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[X * dst_tensor_shape.y + Y], 1);
//			int idx = nnzg_per_hw[X * dst_tensor_shape.y + Y]++;
			nzg_val[idx] = (Z * M_GROUP + 1);
#else
			nzg_val[(Z * M_GROUP + 1)] = 1;
#endif
		}
#endif


	}
	{
		FLT4 res = TO_FLT4(r200) + biases_buffer[Z * M_GROUP + 2];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		res = select((float4)(0.f), res, nnzg_threshold < fabs(res));
		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 2)), res);

#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[X * dst_tensor_shape.y + Y], 1);
//			int idx = nnzg_per_hw[X * dst_tensor_shape.y + Y]++;
			nzg_val[idx] = (Z * M_GROUP + 2);
#else
			nzg_val[(Z * M_GROUP + 2)] = 1;
#endif
		}
#endif


	}
	{
		FLT4 res = TO_FLT4(r300) + biases_buffer[Z * M_GROUP + 3];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		res = select((float4)(0.f), res, nnzg_threshold < fabs(res));
		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 3)), res);

#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[X * dst_tensor_shape.y + Y], 1);
//			int idx = nnzg_per_hw[X * dst_tensor_shape.y + Y]++;
			nzg_val[idx] = (Z * M_GROUP + 3);
#else
			nzg_val[(Z * M_GROUP + 3)] = 1;
#endif
		}
#endif


	}
}
