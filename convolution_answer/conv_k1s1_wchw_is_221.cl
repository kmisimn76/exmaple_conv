#include "kernels/spie_header_cl.h"

#define M_GROUP 4

// conv k1s1 221
__kernel void cl_conv1x1_kernel(
		__read_only  image2d_t	src_tensor,
		__write_only image2d_t	dst_tensor,
#ifndef USE_IMAGE_WEIGHT
		__global 	 float4*	weights_buffer,
#else
		__read_only	 image2d_t	weights_buffer,
#endif
		__global 	 float*		biases_buffer,
		__global	 ushort*		if_idx,
		__global	 int* nnzg_per_hw,
					 int4		src_tensor_shape,	// W_in,  H_in,	 M/4, C/4
					 int4		dst_tensor_shape,	// W_out, H_out, M/4, C/4
					 int4 		kernel_args,
					 int		relu,
					 float		nnzg_threshold
#ifdef MAKE_LUT_IN_KERNEL
					 ,
		__global     ushort*    if_idx,
		__global     int*    nnzg_per_hw,
		__global     int*    nzn_per_group
#endif
				)
{
	int Y = get_global_id(0) * 1;
	int X = get_global_id(1) * 1;
  	int Z = get_global_id(2) * 1;

	if (X >= (dst_tensor_shape.x >> 3) || Y >= (dst_tensor_shape.y >> 1) || Z >= dst_tensor_shape.z) {
		return;
	}

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r100 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r200 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r300 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r001 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r101 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r201 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r301 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);

	ACCUM_FLT4 r010 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r110 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r210 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r310 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// x0, x1, x2, x3
	ACCUM_FLT4 r011 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r111 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r211 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);
	ACCUM_FLT4 r311 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);


	__global float4* weights_cache;
//	__global float4* filters_loc = weights_buffer + Z * M_GROUP * dst_tensor_shape.w;	// MCm4
#ifndef USE_IMAGE_WEIGHT
	__global float4* filters_loc = weights_buffer + Z * 4 * src_tensor_shape.w;
#endif


#ifdef IS_CONV_RA
	int nnzg = nnzg_per_hw[X * (src_tensor_shape.y >> 1) + Y];
//if(X==1 && Y==0 && Z==0)
//printf("aaa(%d) %d\n", src_tensor_shape.y >> 1, nnzg);
	__global ushort* nzg_idx = if_idx + (X * (src_tensor_shape.y >> 1) + Y) * kernel_args.w;
#endif

	int xc0 = X * 2 + 0;
	int xc1 = X * 2 + 1;
	int yc0 = Y * 2 + 0;
	int yc1 = Y * 2 + 1;

#ifdef IS_CONV_RA
	for(int ci=0; ci < nnzg; ci++) {	// random access (nnzg loop)
		ushort s = nzg_idx[ci];
#else
	for(int s=0; s < kernel_args.w; s++) {	// branch (n_group loop)
		if(nzg_idx[s] != 1) continue;
#endif

		float4 src00, src10, src01, src11;
		//weights_cache = filters_loc + s;
#ifndef USE_IMAGE_WEIGHT
		float4 f0 = filters_loc[s];
#else
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(s, Z));
#endif

    	src00 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc0) * kernel_args.w + s));
    	src01 = read_imagef(src_tensor, smp_zero, (int2)((yc1), (xc0) * kernel_args.w + s));
    	src10 = read_imagef(src_tensor, smp_zero, (int2)((yc0), (xc1) * kernel_args.w + s));
    	src11 = read_imagef(src_tensor, smp_zero, (int2)((yc1), (xc1) * kernel_args.w + s));

		r000 += f0.x * src00;	// m=0 / c=0 / x=0~3
		r100 += f0.y * src00; // m=1 / c=0 / x=0~3
		r200 += f0.z * src00;
		r300 += f0.w * src00;

		r001 += f0.x * src10;	
		r101 += f0.y * src10; 
		r201 += f0.z * src10;
		r301 += f0.w * src10;

		r010 += f0.x * src01;	// m=0 / c=0 / x=0~3
		r110 += f0.y * src01; // m=1 / c=0 / x=0~3
		r210 += f0.z * src01;
		r310 += f0.w * src01;

		r011 += f0.x * src11;	// m=0 / c=0 / x=0~3
		r111 += f0.y * src11; // m=1 / c=0 / x=0~3
		r211 += f0.z * src11;
		r311 += f0.w * src11;



	}

	{
		FLT4 res000 = TO_FLT4(r000) + biases_buffer[Z * M_GROUP + 0];
		FLT4 res001 = TO_FLT4(r001) + biases_buffer[Z * M_GROUP + 0];
		FLT4 res010 = TO_FLT4(r010) + biases_buffer[Z * M_GROUP + 0];
		FLT4 res011 = TO_FLT4(r011) + biases_buffer[Z * M_GROUP + 0];
		if(relu == 1) {
			res000 = max(res000, (FLT)(0.0f));
			res001 = max(res001, (FLT)(0.0f));
			res010 = max(res010, (FLT)(0.0f));
			res011 = max(res011, (FLT)(0.0f));
		}
		else if(relu==2) {
			res000 = min(max(res000, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res001 = min(max(res001, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res010 = min(max(res010, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res011 = min(max(res011, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		}
		else if(relu==3) {
			res000 = min(max(res000+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res001 = min(max(res001+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res010 = min(max(res010+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res011 = min(max(res011+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		}
		else if(relu==4) {
			res000 = res000 * (min(max(res000+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res001 = res001 * (min(max(res001+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res010 = res010 * (min(max(res010+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res011 = res011 * (min(max(res011+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
		}

		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 0)), res000);
		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 0)), res001);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 0)), res010);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 0)), res011);
	}
	{
		FLT4 res100 = TO_FLT4(r100) + biases_buffer[Z * M_GROUP + 1];
		FLT4 res101 = TO_FLT4(r101) + biases_buffer[Z * M_GROUP + 1];
		FLT4 res110 = TO_FLT4(r110) + biases_buffer[Z * M_GROUP + 1];
		FLT4 res111 = TO_FLT4(r111) + biases_buffer[Z * M_GROUP + 1];
		if(relu == 1) {
			res100 = max(res100, (FLT)(0.0f));
			res101 = max(res101, (FLT)(0.0f));
			res110 = max(res110, (FLT)(0.0f));
			res111 = max(res111, (FLT)(0.0f));
		}
		else if(relu==2) {
			res100 = min(max(res100, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res101 = min(max(res101, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res110 = min(max(res110, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res111 = min(max(res111, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		}
		else if(relu==3) {
			res100 = min(max(res100+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res101 = min(max(res101+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res110 = min(max(res110+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res111 = min(max(res111+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		}
		else if(relu==4) {
			res100 = res100 * (min(max(res100+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res101 = res101 * (min(max(res101+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res110 = res110 * (min(max(res110+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res111 = res111 * (min(max(res111+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
		}
		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 1)), res100);
		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 1)), res101);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 1)), res110);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 1)), res111);
	}
	
	{
		FLT4 res200 = TO_FLT4(r200) + biases_buffer[Z * M_GROUP + 2];
		FLT4 res201 = TO_FLT4(r201) + biases_buffer[Z * M_GROUP + 2];
		FLT4 res210 = TO_FLT4(r210) + biases_buffer[Z * M_GROUP + 2];
		FLT4 res211 = TO_FLT4(r211) + biases_buffer[Z * M_GROUP + 2];
		if(relu == 1) {
			res200 = max(res200, (FLT)(0.0f));
			res201 = max(res201, (FLT)(0.0f));
			res210 = max(res210, (FLT)(0.0f));
			res211 = max(res211, (FLT)(0.0f));
		}
		else if(relu==2) {
			res200 = min(max(res200, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res201 = min(max(res201, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res210 = min(max(res210, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res211 = min(max(res211, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		}
		else if(relu==3) {
			res200 = min(max(res200+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res201 = min(max(res201+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res210 = min(max(res210+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res211 = min(max(res211+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		}
		else if(relu==4) {
			res200 = res200 * (min(max(res200+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res201 = res201 * (min(max(res201+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res210 = res210 * (min(max(res210+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res211 = res211 * (min(max(res211+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
		}
		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 2)), res200);
		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 2)), res201);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 2)), res210);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 2)), res211);
	}
	{
		FLT4 res300 = TO_FLT4(r300) + biases_buffer[Z * M_GROUP + 3];
		FLT4 res301 = TO_FLT4(r301) + biases_buffer[Z * M_GROUP + 3];
		FLT4 res310 = TO_FLT4(r310) + biases_buffer[Z * M_GROUP + 3];
		FLT4 res311 = TO_FLT4(r311) + biases_buffer[Z * M_GROUP + 3];
		if(relu == 1) {
			res300 = max(res300, (FLT)(0.0f));
			res301 = max(res301, (FLT)(0.0f));
			res310 = max(res310, (FLT)(0.0f));
			res311 = max(res311, (FLT)(0.0f));
		}
		else if(relu==2) {
			res300 = min(max(res300, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res301 = min(max(res301, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res310 = min(max(res310, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
			res311 = min(max(res311, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		}
		else if(relu==3) {
			res300 = min(max(res300+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res301 = min(max(res301+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res310 = min(max(res310+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
			res311 = min(max(res311+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		}
		else if(relu==4) {
			res300 = res300 * (min(max(res300+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res301 = res301 * (min(max(res301+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res310 = res310 * (min(max(res310+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
			res311 = res311 * (min(max(res311+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish
		}

		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 3)), res300);
		write_imagef(dst_tensor, (int2)(Y * 2 + 0, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 3)), res301);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 0) * kernel_args.z + (Z * M_GROUP + 3)), res310);
		write_imagef(dst_tensor, (int2)(Y * 2 + 1, (X * 2 + 1) * kernel_args.z + (Z * M_GROUP + 3)), res311);
	}
	
}
