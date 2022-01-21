#pragma OPENCL SELECT_ROUNDING_MODE rte
//#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
//#define ARM_DOT(x, y, val) val += arm_dot((x), (y));
#define ARM_DOT(a, b, c)        \
   ({                                                   \
         c += (int)(a).s0 * (b).s0; \
         c += (int)(a).s1 * (b).s1; \
         c += (int)(a).s2 * (b).s2; \
         c += (int)(a).s3 * (b).s3; \
   })
__constant sampler_t smp_edge = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef struct __attribute__((packed))  _Param {
        int layer_type;
        int ch_in;
        int dim_in;
        int ch_out;
        int dim_out;
        int pad;
        int in_id;
        int in_id2;
        int relu;
} Param;
typedef struct __attribute__((packed)) shape_ {
        int c;
        int h;
        int w;
        int n;
} shape;
enum TYPE {
        CONV_K7S2,      // 0
        CONV_K1S1,      // 1
        CONV_K1S2,      // 2
        CONV_K3S1,      // 3
        CONV_K5S1,      // 4
        MAX_POOL_K3S1,  // 5
        MAX_POOL_K3S2,  // 6
        AVG_POOL_K7,    // 7
        LRN,    // 8
        FC,             // 9
        CONCAT, // 10
        ELTWISE_ADD     // 11
};
//#define ACCUM_FLT4 float4
//#define FLT float
//#define FLT4 float4
//#define TO_FLT4 convert_float4
#define ACCUM_FLT4 float4
#define FLT float
#define FLT4 float4
#define FLT16 float16
#define TO_FLT4 convert_float4
#define TO_ACCUM_FLT4 convert_float4
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
	int X = get_global_id(0) * 1;	// E-axis (=W)
	int Y = get_global_id(1) * 1;	// F-axis (=H)
  	int Z = get_global_id(2) * 1;	// M-axis (M/4)

//	if(X==0&&Y==0&&Z==0)
//		printf("%d %d %d\n", get_global_size(0), get_global_size(1) , get_global_size(2)) ;
	if (X >= dst_tensor_shape.x || Y >= dst_tensor_shape.y || Z >= dst_tensor_shape.z) {
		return;
	}

#ifdef MAKE_LUT_IN_KERNEL
	 __global ushort* nzg_val = if_idx + (X * dst_tensor_shape.y + Y) * src_tensor_shape.z;
	 float iszero;
#endif

	ACCUM_FLT4 r000 = (ACCUM_FLT4)(0.f, 0.f, 0.f, 0.f);	// ACCUM_FLT4 == float4 (TF Lite notation)
	__global float4* weights_cache;	// 4 filter values for a specific channel in a vector
#ifndef USE_IMAGE_WEIGHT
	__global float4* filters_loc = weights_buffer + Z * 4 * src_tensor_shape.w;
#endif
	int xc0 = X * 1;
	int yc0 = Y * 1;
	int s = 0;
	int c = 0;
	//for(int i=0;i<240;i++) {	// reduction loop for entire C
    do {
		float4 src00;
#ifndef USE_IMAGE_WEIGHT
    	src00 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + s));	// HCWc4 format
		s += 1;
		weights_cache = filters_loc;
		r000 += weights_cache[0] * src00.x;	// src00.x is widen; 4 filter values for a channel are calculated
		r000 += weights_cache[1] * src00.y;	// another channel
		r000 += weights_cache[2] * src00.z;	// another channel
		r000 += weights_cache[3] * src00.w;	// another channel
		filters_loc += 4;
#else
//			unsigned long rand = ((14895+(get_global_id(0)*get_global_size(1)+get_global_id(1))*get_global_size(2)+get_global_id(2)+s) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
//			unsigned int ns = (rand >> 16) % src_tensor_shape.w;
		//unsigned int ns = s;
    	src00 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + s));	// HCWc4 format
	    //	src00 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + ns));	// HCWc4 format
		s += 1;
		float4 f0, f1, f2, f3;
		f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z));
		f1 = read_imagef(weights_buffer, smp_zero, (int2)(c+1, Z));
		f2 = read_imagef(weights_buffer, smp_zero, (int2)(c+2, Z));
		f3 = read_imagef(weights_buffer, smp_zero, (int2)(c+3, Z));
//			r000 += ns * src00;
		r000 += f0 * src00.x;
		r000 += f1 * src00.y;
		r000 += f2 * src00.z;
		r000 += f3 * src00.w;
		c += 4;
//		r000 += f0 + f1 + f2 + f3;
/*
    	src00 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + s));	// HCWc4 format
		float4 src01=(float4)(0.f, 0.f, 0.f, 0.f);
    	src01 = read_imagef(src_tensor, smp_zero, (int2)((xc0), (yc0) * src_tensor_shape.w + s+1));	// HCWc4 format
		s += 2;
		float4 f0 = read_imagef(weights_buffer, smp_zero, (int2)(c, Z));
		float4 f1 = read_imagef(weights_buffer, smp_zero, (int2)(c+1, Z));
		float4 f2 = read_imagef(weights_buffer, smp_zero, (int2)(c+2, Z));
		float4 f3 = read_imagef(weights_buffer, smp_zero, (int2)(c+3, Z));
		float4 f4 = read_imagef(weights_buffer, smp_zero, (int2)(c+4, Z));
		float4 f5 = read_imagef(weights_buffer, smp_zero, (int2)(c+5, Z));
		float4 f6 = read_imagef(weights_buffer, smp_zero, (int2)(c+6, Z));
		float4 f7 = read_imagef(weights_buffer, smp_zero, (int2)(c+7, Z));
//		printf("%lf %lf %lf %lf\n", f0.x, f1.y, f2.z, f3.w);
		r000 += (f0 * src00.x + f1 * src00.y + f2 * src00.z + f3 * src00.w);
		r000 += (f4 * src01.x + f5 * src01.y + f6 * src01.z + f7 * src01.w);
		c += 8;*/
#endif
	} while (s < src_tensor_shape.w);	// loop over C/4

	weights_cache = biases_buffer + Z;
	{
		FLT4 bias_val = TO_FLT4(weights_cache[0]); 
		FLT4 res = TO_FLT4(r000) + bias_val;
	if(X==0 && Y==0 && Z==0)
//		printf("1x1 conv output: %lf %d\n", res.x, relu);
		if(relu==1) res = max(res, (FLT)(0.0f));
		else if(relu==2) res = min(max(res, (FLT)(0.0f)), (FLT)(6.0f)); //ReLU 6
		else if(relu==3) res = min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f); //hard sigmoid
		else if(relu==4) res = res * (min(max(res+(FLT)(3.0f), (FLT)(0.0f)), (FLT)(6.0f)) / (FLT)(6.0f)); //hard swish

		res = select((float4)(0.f), res, nnzg_threshold < fabs(res));
		write_imagef(dst_tensor, (int2)(((X + 0)), ((Y + 0)) * dst_tensor_shape.z + (Z + 0)), res);

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
