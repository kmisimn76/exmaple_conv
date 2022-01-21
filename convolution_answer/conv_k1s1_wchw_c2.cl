#include "kernels/spie_header_cl.h"

#define M_GROUP 4

// conv k1s1 111
__kernel void cl_conv1x1_kernel(
		__read_only  image2d_t	src_tensor,
		__write_only image2d_t	dst_tensor,
		__read_only	 image2d_t	weights_buffer,
		__global 	 float*		biases_buffer,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,		// K,    stride, M,   C
					 int 		relu )
{
	int Y = get_global_id(0) * 1;
	int X = get_global_id(1) * 1;
  	int Z = get_global_id(2) * 2;

	if (X >= (dst_tensor_shape.x >> 2) || Y >= dst_tensor_shape.y || Z >= (dst_tensor_shape.z<<2)) {
		return;
	}

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3

	ACCUM_FLT4 r001 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	__global float4* weights_cache;

	int xc0 = X * 1;
	int yc0 = Y * 1;

	int s = 0;
	int c = 0;
	do {
		float4 src0_00;
		float4 src1_00;
		float4 src2_00;
		float4 src3_00;

    	src0_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s));
    	src1_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+1));
    	src2_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+2));
    	src3_00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s+3));
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z));
		float4 f1 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z+1));

		r000 += f0.x * src0_00;
		r000 += f0.y * src1_00;
		r000 += f0.z * src2_00;
		r000 += f0.w * src3_00;
		r001 += f1.x * src0_00;
		r001 += f1.y * src1_00;
		r001 += f1.z * src2_00;
		r001 += f1.w * src3_00;

		s += 4;
		c += 1;
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
	}
	if(Z+1 >= (dst_tensor_shape.z<<2)) return;
	{
		FLT4 res = TO_FLT4(r001) + biases_buffer[Z+1];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z+1)), res);
	}
}
