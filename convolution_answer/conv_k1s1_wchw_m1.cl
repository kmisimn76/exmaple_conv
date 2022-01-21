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

//	if(X==0 && Y==0 && Z==0)
//		printf("its here! %d %d %d\n", dst_tensor_shape.x, dst_tensor_shape.y, dst_tensor_shape.z);
	if (X >= (dst_tensor_shape.x >> 2) || Y >= dst_tensor_shape.y || Z >= (dst_tensor_shape.z<<2)) {
		return;
	}

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	__global float4* weights_cache;
#ifndef USE_IMAGE_WEIGHT
	__global float4* filters_loc = weights_buffer + Z * src_tensor_shape.w;
#endif

	int xc0 = X * 1;
	int yc0 = Y * 1;

	int s = 0;
	int c = 0;
	do {
		float4 src00;
		float4 src01;
		float4 src02;
		float4 src03;

    	src00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s));
    	src01 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+1));
    	src02 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+2));
    	src03 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+3));
#ifndef USE_IMAGE_WEIGHT
		float4 f0 = filters_loc[0];
		filters_loc += 1;
#else
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z));
		c += 1;
#endif

		r000 += f0.x * src00;	// m=0 / c=0 / x=0~3
		r000 += f0.y * src01;	// m=0 / c=0 / x=0~3
		r000 += f0.z * src02;	// m=0 / c=0 / x=0~3
		r000 += f0.w * src03;	// m=0 / c=0 / x=0~3
    	
		s += 4;
	} while (s < kernel_args.w);
//if(X==0&&Y==0&&Z==0)
//	printf("%d,%d,%d %d %lf %d\n",dst_tensor_shape.x ,dst_tensor_shape.y,dst_tensor_shape.z, kernel_args.z, r000.x, kernel_args.w);

//	weights_cache = biases_buffer + Z;
//	if(Z + 0 >= tensor_shape.z) return;	// no need for this kernel 111
	{
		FLT4 res = TO_FLT4(r000) + biases_buffer[Z];
	//if(X==0 && Y==0 && Z==0 && kernel_args.w==6 && kernel_args.z==18)
	if(X==0 && Y==0 && Z==0)
//		printf("1x1 conv output: %lf %d\n", res.x, relu);
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z)), res);
#ifdef MAKE_LUT_IN_KERNEL
		iszero = dot(res, (float4)(1.f, 1.f, 1.f, 1.f));
		if(iszero != 0.f) {
#ifdef IS_CONV_RA
			int idx = atomic_add(&nnzg_per_hw[X * dst_tensor_shape.y + Y], 1);
			//int idx = nnzg_per_hw[X * dst_tensor_shape.y + Y]++;
			nzg_val[idx] = Z;
#else
			nzg_val[Z] = 1;
#endif
		}
#endif


	}
}
