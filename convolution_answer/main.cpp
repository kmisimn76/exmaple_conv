#include <iostream>
#include <fstream>
#include <CL/cl.hpp>
#include <cstdlib>
#include <cmath>
#include <time.h>
#define RAND_NORM() (-1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/2.0f)))

void getConvolutionGold(const float*,const float*,const float*,const int,const int,const int,const int,const int,float*);
void getKernelError(float*, float*, int);

// apparently OpenCL only likes arrays ...
#define WH_in 56 //input width/height: WH+RS-1
#define WH 56	//output width/height
#define RS 3	//filter size
#define C 256		//input channel
#define K 256		//output channel
float input[WH_in*WH_in*C]; // CHW == input[C][WH_in][WH_in]
float weight[RS*RS*C*K];	 // KCRS == weight[K][C][RS][RS]
float bias  [K];
float output[WH*WH*K];		 // KHW == output[K][WH][WH}
float output_gold[WH*WH*K];

int main()
{
	    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({default_device});

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

    // calculates for each element; C = A + B
	std::ifstream ifs("conv.cl");
	std::string kernel_code( (std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()    ) );
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

	// Set random input
	for(int i=0;i<WH_in*WH_in*C;i++) input [i] = RAND_NORM();
	for(int i=0;i<RS*RS*C*K;i++)	 weight[i] = RAND_NORM();
	for(int i=0;i<K;i++)			 bias  [i] = RAND_NORM();

    // create buffers on device (allocate space on GPU)
    cl::Buffer buffer_input (context, CL_MEM_READ_WRITE, sizeof(float) * WH_in*WH_in*C);
    cl::Buffer buffer_weight(context, CL_MEM_READ_WRITE, sizeof(float) * RS*RS*C*K);
    cl::Buffer buffer_bias  (context, CL_MEM_READ_WRITE, sizeof(float) * K);
    cl::Buffer buffer_output(context, CL_MEM_READ_ONLY,  sizeof(float) * WH*WH*K);

    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, default_device);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_input , CL_TRUE, 0, sizeof(float)*WH_in*WH_in*C, input);
    queue.enqueueWriteBuffer(buffer_weight, CL_TRUE, 0, sizeof(float)*RS*RS*C*K, weight);
    queue.enqueueWriteBuffer(buffer_bias  , CL_TRUE, 0, sizeof(float)*K, bias);

	// Set Kernel
	cl::Kernel kernel(program, "convolution");
	kernel.setArg(0, buffer_input);
	kernel.setArg(1, buffer_weight);
	kernel.setArg(2, buffer_bias);
	kernel.setArg(3, buffer_output);
	kernel.setArg(4, WH_in);
	kernel.setArg(5, WH);
	kernel.setArg(6, RS);
	kernel.setArg(7, C);
	kernel.setArg(8, K);

	struct timespec start, end;
	double gpu_kernel_time;
	double cpu_time;

	// Run Kernel
	clock_gettime(CLOCK_MONOTONIC, &start);
	//start
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(K, WH, WH), cl::NullRange);
	queue.finish();
	//end
	clock_gettime(CLOCK_MONOTONIC, &end);
	gpu_kernel_time = (end.tv_sec - start.tv_sec) * 1000.0
                  + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    // read result from GPU to here
    queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float)*WH*WH*K, output);

	// CPU Excution
	clock_gettime(CLOCK_MONOTONIC, &start);
	//start
	getConvolutionGold(input, weight, bias, WH_in, WH, RS, C, K, output_gold);
	//end
	clock_gettime(CLOCK_MONOTONIC, &end);
	cpu_time = (end.tv_sec - start.tv_sec) * 1000.0
                  + (end.tv_nsec - start.tv_nsec) / 1000000.0;

	// Calcuate error
	getKernelError(output, output_gold, WH*WH*K);
	std::cout << "GPU time: " << gpu_kernel_time << " ms, CPU time: " << cpu_time << " ms\n";

    return 0;
}

void getConvolutionGold(const float* input,
						const float* weight,
						const float* bias,
						const int _WH_in,
						const int _WH,
						const int _RS,
						const int _C,
						const int _K,
						float* output_gold)
{
	for (int k=0;k<_K;k++) {
		for (int h=0;h<_WH;h++) {
			for (int w=0;w<_WH;w++) {
				int idx_output = k*_WH*_WH + h*_WH + w;
				output_gold[idx_output] = bias[k];
				for (int r=0;r<_RS;r++) {
					for (int s=0;s<_RS;s++) {
						for (int c=0;c<_C;c++) {
							int idx_input = c*_WH_in*_WH_in + (h+r)*_WH_in + (w+s);
							int idx_weight = k*_C*_RS*_RS + c*_RS*_RS + r*_RS + s;
							output_gold[idx_output] += input[idx_input] * weight[idx_weight];
						}
					}
				}
			}
		}
	}
}

void getKernelError(float* A, float* B, int N)
{
	float err = 0.0, sum = 0.0;
	for(int i=0;i<N;i++) { err += abs(A[i]-B[i]); sum += abs(B[i]); }
	std::cout << "Kernel Ouptut Error: " << err << "(" << err/sum*100.0 << "%)\n";
}
