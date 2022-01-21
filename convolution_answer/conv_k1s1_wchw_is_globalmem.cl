#include "kernels/spie_header_cl.h"

#define M_GROUP 4

// conv k1s1 111
__kernel void cl_conv1x1_kernel(
		__read_only  image2d_t	src_tensor,
		__write_only image2d_t	dst_tensor,
		__global 	 float4*	weights_buffer,
		__global 	 float*		biases_buffer,
		__global	 ushort*		if_idx,
		__global	 ushort*	nnzg_per_hw,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,
					 int		relu)		// nnzg,  stride, M,   C
{
	int Y = get_global_id(0) * 1;
	int X = get_global_id(1) * 1;
  	int Z = get_global_id(2) * 1;

//	if (X >= (dst_tensor_shape.x >> 2) || Y >= dst_tensor_shape.y || Z >= kernel_args.z) {
	if (X >= (dst_tensor_shape.x >> 2) || Y >= dst_tensor_shape.y || Z >= dst_tensor_shape.z) {
		return;
	}

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r100 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r200 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r300 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	__global float4* weights_cache;
	__global float4* filters_loc = weights_buffer + Z * M_GROUP * dst_tensor_shape.w;	// MCm4


#ifdef IS_CONV_RA
	int nnzg = nnzg_per_hw[X * src_tensor_shape.y + Y];
#endif
	__global ushort* nzg_idx = if_idx + (X * src_tensor_shape.y + Y) * kernel_args.w;

	int xc0 = X * 1;
	int yc0 = Y * 1;

#ifdef IS_CONV_RA
	for(int ci=0; ci < nnzg; ci++) {	// random access (nnzg loop)
		ushort s = nzg_idx[ci];
#else
	for(int s=0; s < kernel_args.w; s++) {	// branch (n_group loop)
		if(nzg_idx[s] != 1) continue;
#endif

		float4 src00;
		weights_cache = filters_loc + s;

    	src00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s));

		r000 += weights_cache[0].x * src00;	// m=0 / c=0 / x=0~3
		r100 += weights_cache[0].y * src00; // m=1 / c=0 / x=0~3
		r200 += weights_cache[0].z * src00;
		r300 += weights_cache[0].w * src00;
	}

	{
		FLT4 res = TO_FLT4(r000) + biases_buffer[Z * M_GROUP + 0];
		if(relu == 1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 0)), res);
	}
	{
		FLT4 res = TO_FLT4(r100) + biases_buffer[Z * M_GROUP + 1];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 1)), res);
	}
	{
		FLT4 res = TO_FLT4(r200) + biases_buffer[Z * M_GROUP + 2];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 2)), res);
	}
	{
		FLT4 res = TO_FLT4(r300) + biases_buffer[Z * M_GROUP + 3];
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		write_imagef(dst_tensor, (int2)(Y, X * kernel_args.z + (Z * M_GROUP + 3)), res);
	}
}
