//************************************************************
// Demo OpenCL application to compute a simple vector addition
// computation between 2 arrays on the GPU
// ************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
//
// OpenCL source code
const char* OpenCLSource[] = {
"__kernel void VectorAdd(__global uchar * yuv, __global uchar * rgb,int W,int H)"\
"{"\
" unsigned int i =0;"\
" unsigned int j =0;"\
" uchar y,u,v,r,g,b;"\
" y=yuv[j*W+i];"\
" u=yuv[W*H+(i+W*j)/4];"\
" v=yuv[W*H+(i+W*j)/4+1];"\
" r=y+1.403*(v-128);"\
" g=y-0.343*(u-128)-0.714*(v-128);"\
" b=y+1.770*(u-128);"\
" rgb[j*3+i]=r;"\
" rgb[j*3+i+1]=g;"\
" rgb[j*3+i+2]=b;"\
"}"\
};

#define HEIGHT 256
#define WIDTH 256
#define SIZE HEIGHT*WIDTH*3
// Main function
// ************************************************************
char cpu_yuv[WIDTH*HEIGHT*3/2];
char cpu_rgb[WIDTH*HEIGHT*3];

int main(int argc, char **argv)
{	

	int width=WIDTH;
	int height=HEIGHT;
	cl_int status;

     //Get an OpenCL platform
     cl_platform_id cpPlatform;
     clGetPlatformIDs(1, &cpPlatform, NULL);
     // Get a GPU device
     cl_device_id cdDevice;
     clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
     char cBuffer[1024];
     clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
     printf("CL_DEVICE_NAME: %s\n", cBuffer);
     clGetDeviceInfo(cdDevice, CL_DRIVER_VERSION, sizeof(cBuffer), &cBuffer, NULL);
     printf("CL_DRIVER_VERSION: %s\n\n", cBuffer);
     // Create a context to run OpenCL enabled GPU
     cl_context GPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);     
     // Create a command-queue on the GPU device
     cl_command_queue cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, NULL);
     // Allocate GPU memory for source vectors AND initialize from CPU memory
     cl_mem gpu_yuv = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
     CL_MEM_COPY_HOST_PTR, sizeof(char) * SIZE, cpu_yuv, NULL);
     // Allocate output memory on GPU
     cl_mem gpu_rgb = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY,
     sizeof(char) * SIZE, NULL, NULL);
     // Create OpenCL program with source code
     cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 1, OpenCLSource, NULL, NULL);
     // Build the program (OpenCL JIT compilation)
  	status = clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
 	if(status != 0)
 		{
 		printf("clBuild failed:%d\n", status);
 		char tbuf[0x10000];
 		clGetProgramBuildInfo(OpenCLProgram, 0, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
 		printf("\n%s\n", tbuf);
 		return -1;
 		}
     // Create a handle to the compiled OpenCL function (Kernel)
     cl_kernel OpenCLyuv2rgb = clCreateKernel(OpenCLProgram, "yuv2rgb", NULL);
     // In the next step we associate the GPU memory with the Kernel arguments
     clSetKernelArg(OpenCLyuv2rgb, 0, sizeof(cl_mem), (void*)&gpu_yuv);
     clSetKernelArg(OpenCLyuv2rgb, 1, sizeof(cl_mem), (void*)&gpu_rgb);
     clSetKernelArg(OpenCLyuv2rgb, 2, sizeof(cl_int),  (void *)&width);
     clSetKernelArg(OpenCLyuv2rgb, 3, sizeof(cl_int),  (void *)&height);
     
     //create event
     cl_event event = clCreateUserEvent(GPUContext, NULL);
     
     // Launch the Kernel on the GPU

 	size_t globalThreads[] = {width,height};
 	size_t localThreads[] = {16, 16}; // localx*localy应该是64的倍数
     clEnqueueNDRangeKernel(cqCommandQueue, OpenCLyuv2rgb, 2, NULL, globalThreads, localThreads, 0, NULL, &event);
     // Copy the output in GPU memory back to CPU memory
     clEnqueueReadBuffer(cqCommandQueue, gpu_rgb, CL_TRUE, 0,
     SIZE * sizeof(char), cpu_rgb, 0, NULL, NULL);
     // Cleanup


     
     clWaitForEvents(1, &event);
     cl_ulong start = 0, end = 0;
     double total_time;     
     
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	  clReleaseKernel(OpenCLyuv2rgb);
	  clReleaseProgram(OpenCLProgram);
	  clReleaseCommandQueue(cqCommandQueue);
	  clReleaseContext(GPUContext);
	  clReleaseMemObject(gpu_rgb);
	  clReleaseMemObject(gpu_yuv);

     total_time = end - start;     
        
//     for( int i =0 ; i < SIZE; i++)
//     {
//     	printf("[%d + %d = %d]\n",HostVector1[i], HostVector2[i], HostOutputVector[i]);
//     }
     
     printf("\nExecution time in milliseconds = %0.3f ms", (total_time / 1000000.0) );
     printf("\nExecution time in seconds = %0.3f s\n\n", ((total_time / 1000000.0))/1000 );          
          
     return 0;
}
