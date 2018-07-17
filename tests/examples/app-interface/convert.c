//0e_multidim.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define MAX_RESOLUTION_NUM  16
typedef struct{
unsigned int width;
unsigned int height;
unsigned int total_width;
unsigned int total_height;
}def_resolution;

def_resolution resolution[MAX_RESOLUTION_NUM];

#define LOCAL_ITEM_SIZE  16

#define NUM_ELEMENTS_X 1920
#define NUM_ELEMENTS_Y 1120
#define NUM_ELEMENTS (NUM_ELEMENTS_X*NUM_ELEMENTS_Y*3)

/* Try setting this to 1 to verify that get_global_id()
 * is equivalent to a formula written using get_group_id(),
 * get_local_size() and get_local_id().
 */
#define VERBOSE_GLOBAL_ID 0

/* A kernel which sets all elements of an array to the 1D ID of the work group */
//const char * source_str  =	"__kernel void yuvscale(__global uchar * dest, __global uchar * src,int srcwidth, int srcheight)"
//							"{"
//							" unsigned int i =get_global_id(0);"
//							" unsigned int j =get_global_id(1);"
//							" unsigned int destwidth =get_global_size(0);"
//							" unsigned int destheight =get_global_size(1);"
//							"float w_scale_rate = (float)srcwidth / destwidth;"
//							"float h_scale_rate = (float)srcheight / destheight;"
//							"float i_scale = h_scale_rate * i;"
//							"float j_scale = w_scale_rate * j;"
//							"int srci = i_scale;"
//							"int srcj = j_scale;"
//							"float u = i_scale - srci;"
//							"float v = j_scale - srcj;"
//							"int offset=srcj*srcwidth+srci;"
//							"dest[j*destwidth+i]=((1-u)*(1-v)*src[offset]+(1-u)*v*src[offset+srcwidth]+u*(1-v)*src[offset+1]+u*v*src[offset+srcwidth+1]);"
//							"offset=srcwidth*srcheight+(srcj>>1)*srcwidth+srci-srci%2;"
//							"unsigned int tmp1=destwidth*destheight+(j>>1)*destwidth+i-i%2;"
//							" dest[tmp1]=((1-u)*(1-v)*src[offset]+(1-u)*v*src[offset+(srcwidth>>1)]+u*(1-v)*src[offset+1-(srci+1)%2+srci%2]+u*v*src[offset+(srcwidth>>1)+1-(srci+1)%2+srci%2]);"
//							" dest[tmp1+1]=((1-u)*(1-v)*src[offset+1]+(1-u)*v*src[offset+(srcwidth>>1)+1]+u*(1-v)*src[offset+1-(srci+1)%2+srci%2+1]+u*v*src[offset+(srcwidth>>1)+1-(srci+1)%2+srci%2+1]);"
//							"}";








cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_uint num_devices;
cl_uint num_platforms;
cl_int ret;
cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel,kernel0;
cl_mem gpu_rgb;
cl_mem gpu_yuv,gpu_yuv0;
cl_mem gpu_arg;
char * gpu_cpu_rgb;
char *gpu_cpu_yuv;
unsigned int items_width,items_height;

cl_int gpu_width,gpu_height;

cl_event event;



cl_program CreateProgrambytearray(cl_context context, cl_device_id device, char *byte)
{
    cl_int errNum = 0;
    cl_program program;


    program = clCreateProgramWithSource(context, 1, (const char **)&byte, NULL, NULL);


    if (!program)
    {
        fprintf(stderr, "Failed to create program from source\n");
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (CL_SUCCESS != errNum)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Error when building program:\n%s\n", buildLog);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}


cl_program CreateProgram(cl_context context, cl_device_id device, const char *kernelFileName)
{
    cl_int errNum = 0;
    cl_program program;

    FILE *file = fopen(kernelFileName, "r");
    if (!file)
    {
        fprintf(stderr, "Failed to open kernel file\n");
        return NULL;
    }

    struct stat st;
    stat(kernelFileName, &st);
    size_t fileSize = st.st_size;

    if (0 == fileSize)
    {
        fprintf(stderr, "Kernel source file was empty\n");
        return NULL;
    }

    char *fileBuffer = malloc(fileSize);

    size_t bytesRead = fread(fileBuffer, sizeof(char), fileSize, file);

    fclose(file);

    if (bytesRead != fileSize)
    {
        fprintf(stderr, "Failed to read complete kernel source file\n");
        free(fileBuffer);
        return NULL;
    }

    program = clCreateProgramWithSource(context, 1, (const char **)&fileBuffer, NULL, NULL);

    free(fileBuffer);

    if (!program)
    {
        fprintf(stderr, "Failed to create program from source\n");
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (CL_SUCCESS != errNum)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Error when building program:\n%s\n", buildLog);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}




void printf_callback( const char *buffer, size_t len, size_t complete, void *user_data )
{
    printf( "%.*s", len, buffer );
}

#define MULTIPLE(number,multiple) number+((number%multiple)==0?0:(multiple-(number%multiple)))


#include "convert.h"
int convert_init(int width,int height,float ratio) {



    /* Get platform and device information */

	memset(resolution,0,sizeof(resolution));

	//ratio=0.7;
	resolution[0].width=width;
	resolution[1].width=640;
	resolution[1].total_width=0;
	resolution[0].height=height;
	resolution[1].height=resolution[0].height/(resolution[0].width/resolution[1].width);
	resolution[1].total_height=0;

	for(int i=2;i<MAX_RESOLUTION_NUM;i++){
		resolution[i].width=resolution[i-1].width*ratio;
		resolution[i].total_width=0;
		resolution[i].height=resolution[i-1].height*ratio;
		resolution[i].total_height=resolution[i-1].height+resolution[i-1].total_height;
	}

	resolution[0].width=MULTIPLE(resolution[0].width,LOCAL_ITEM_SIZE);
	resolution[0].height=MULTIPLE(resolution[0].height,LOCAL_ITEM_SIZE);

	resolution[0].total_width=items_width=resolution[1].width;
	resolution[0].total_height=items_height=MULTIPLE(resolution[MAX_RESOLUTION_NUM-1].total_height,LOCAL_ITEM_SIZE);



	for(int i=0;i<MAX_RESOLUTION_NUM;i++){

		printf("\nresolution[%d].width=%u \n",i,resolution[i].width);
		printf("resolution[%d].total_width=%u \n",i,resolution[i].total_width);
		printf("resolution[%d].height=%u \n",i,resolution[i].height);
		printf("resolution[%d].total_height=%u \n",i,resolution[i].total_height);
	}


    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);


    cl_context_properties properties[] =
    {
        /* Enable a printf callback function for this context. */
        CL_PRINTF_CALLBACK_ARM,   (cl_context_properties) printf_callback,

        /* Request a minimum printf buffer size of 4MiB for devices in the
           context that support this extension. */
        CL_PRINTF_BUFFERSIZE_ARM, (cl_context_properties) 0x100000,

        CL_CONTEXT_PLATFORM,      (cl_context_properties) platform_id,
        0
    };

    /* Create an OpenCL context */
    context = clCreateContext( properties, 1, &device_id, NULL, NULL, &ret);


    /* Create a command queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Allocate host array. Note that the elements are zeroed. */

    /* Print the zeroed array. */
   // printArray(host_arr, NUM_ELEMENTS);

    /* Create device memory buffer */

    /* Create a program from the kernel source */
//    program = clCreateProgramWithSource(context, 1, &source_str, NULL, &ret);
//
//    /* Build the program */
//    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
// 	if(ret != 0)
// 		{
// 		printf("clBuild failed:%d\n", ret);
// 		char tbuf[0x10000];
// 		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
// 		printf("\n%s\n", tbuf);
// 		return -1;
// 		}
   // program=CreateProgram(context,device_id,"convert.cl");
    program=CreateProgrambytearray(context,device_id,convert_cl);






 	{
		/* Create the OpenCL kernel */
		
			kernel0 = clCreateKernel(program, "nv12gbr", &ret);
	 		if(ret!=CL_SUCCESS)
	 			printf("%d clCreateKernel=%d\n",__LINE__,ret);
 		kernel = clCreateKernel(program, "scale", &ret);
 		if(ret!=CL_SUCCESS)
 			printf("%d clCreateKernel=%d\n",__LINE__,ret);




    gpu_yuv0 = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_ALLOC_HOST_PTR,
    		resolution[0].width*resolution[0].height*3 * sizeof(char), NULL, &ret);


    cl_image_format format;
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_RGB;

    /* Allocate memory for the input image that can be accessed by the CPU and GPU. */


 		gpu_yuv = clCreateImage2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &format, resolution[0].width, resolution[0].height, 0, NULL, &ret);
 		if(ret!=CL_SUCCESS)
 			printf("%d clCreateImage2D=%d\n",__LINE__,ret);

 		gpu_rgb = clCreateImage2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, &format, resolution[0].total_width, resolution[0].total_height, 0, NULL, &ret);

 		if(ret!=CL_SUCCESS)
 			printf("%d clCreateImage2D=%d\n",__LINE__,ret);



	    ret = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &gpu_yuv);
		 ret = clSetKernelArg(kernel0, 1, sizeof(cl_mem), &gpu_yuv0);
    /* Set the kernel argument */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_rgb);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_yuv);

		gpu_arg = clCreateBuffer(context, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR,
				sizeof(resolution), resolution, &ret);

	    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_arg);
		
		
		

 	}
	
	



}


#define USEMMAP
int convert(char *cpu_yuv,char *cpu_rgb)
{

   struct timeval  start,end;
    cl_int ret;


	

	
    size_t origin[3] = {0, 0, 0};
     size_t region[3] = {resolution[0].width, resolution[0].height, 1};
     size_t rowPitch;


     gettimeofday(&start,NULL);

  gpu_cpu_yuv=clEnqueueMapBuffer(command_queue,gpu_yuv0,CL_TRUE,CL_MAP_WRITE,0,resolution[0].height*resolution[0].width*3 * sizeof(char),0,0,0,0);
    //printf("\ngpu_cpu_yuv=%p\n",gpu_cpu_yuv);


    memcpy(gpu_cpu_yuv,cpu_yuv, resolution[0].height*resolution[0].width*3/2);
    //ret = clFlush(command_queue);
    //ret = clFinish(command_queue);

    clEnqueueUnmapMemObject(command_queue,gpu_yuv0,gpu_cpu_yuv,0,0,0);





//    gettimeofday(&end,NULL);
//    printf("\ntime %ld ms\n",end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000));
//
//    gettimeofday(&start,NULL);
    /* Execute the OpenCL kernel */
   // printf("items_width=%u, items_height=%u\n",items_width, items_height);
    size_t global_item_size[2] = {resolution[0].width, resolution[0].height};
    size_t local_item_size[2] = {LOCAL_ITEM_SIZE, LOCAL_ITEM_SIZE};
    ret = clEnqueueNDRangeKernel(command_queue, kernel0, 2, NULL, global_item_size, local_item_size, 0, NULL, &event);


//    clWaitForEvents(1, &event);
//    if (ret != CL_SUCCESS) {
//        printf("%s %d OpenCL error executing kernel: %d\n",__FUNCTION__,__LINE__,ret);
//       // goto cleanup;
//    }

//    gettimeofday(&end,NULL);
//    printf("\ntime %ld ms\n",end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000));
//
//    gettimeofday(&start,NULL);

//     gpu_cpu_yuv=clEnqueueMapImage(command_queue,  gpu_yuv, CL_TRUE, CL_MAP_READ, origin, region, &rowPitch, NULL, 0, NULL, NULL, &ret);
//    // printf("gpu_cpu_rgb=%p\n",gpu_cpu_rgb);
//     if (ret != CL_SUCCESS) {
//         printf("%s %d %d\n",__FUNCTION__,__LINE__,ret);
//        // goto cleanup;
//     }
//
//     memcpy(cpu_rgb,gpu_cpu_yuv, resolution[0].width*resolution[0].height*3);
//    // ret = clFlush(command_queue);
//    // ret = clFinish(command_queue);
//
//     clEnqueueUnmapMemObject(command_queue,gpu_rgb,gpu_cpu_rgb,0,0,0);
//
//
//     return 0;

    global_item_size[0] = resolution[0].total_width;
    global_item_size[1] = resolution[0].total_height;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, &event);


//    ret = clFlush(command_queue);
//    ret = clFinish(command_queue);
    clWaitForEvents(1, &event);
    if (ret != CL_SUCCESS) {
        printf("%s %d OpenCL error executing kernel: %d\n",__FUNCTION__,__LINE__,ret);
       // goto cleanup;
    }



//    gettimeofday(&end,NULL);
//
//    printf("time %ld ms\n",end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000));
//
//    gettimeofday(&start,NULL);
    /* Read the results from the device memory buffer back into host array */

    int index=1;
//    origin[0]=resolution[index].total_width;
//    origin[1]=resolution[index].total_height;
//    origin[2]=0;
//
//    size_t newRegion[3] = {resolution[index].width, resolution[index].height, 1};

       size_t newRegion[3] = {resolution[0].total_width, resolution[0].total_height, 1};
#ifdef USEMMAP
    gpu_cpu_rgb=clEnqueueMapImage(command_queue,  gpu_rgb, CL_TRUE, CL_MAP_READ, origin, newRegion, &rowPitch, NULL, 0, NULL, NULL, &ret);
   // printf("gpu_cpu_rgb=%p\n",gpu_cpu_rgb);
    if (ret != CL_SUCCESS) {
        printf("%s %d %d\n",__FUNCTION__,__LINE__,ret);
       // goto cleanup;
    }
 //   memcpy(cpu_rgb,gpu_cpu_rgb,resolution[index].width*resolution[index].height*3);
      memcpy(cpu_rgb,gpu_cpu_rgb,resolution[0].total_width*resolution[0].total_height*3);
   // ret = clFlush(command_queue);
   // ret = clFinish(command_queue);

    clEnqueueUnmapMemObject(command_queue,gpu_rgb,gpu_cpu_rgb,0,0,0);
#else

     ret = clEnqueueReadBuffer(command_queue, gpu_rgb, CL_TRUE, 0, width*height*3 * sizeof(char), cpu_rgb, 0, NULL, NULL);
#endif





     gettimeofday(&end,NULL);
     long int delta=end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000);
    // if(delta>30)
     printf("time %ld ms\n",delta);





    /* Print the fetched array */
   // printArray(host_arr, NUM_ELEMENTS);



}

int convert_deinit()
{

    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(gpu_rgb);
    ret = clReleaseMemObject(gpu_yuv);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}


int main()
{

	convert_init(1920,1080,0.7);

    /* Allocate host array. Note that the elements are zeroed. */
    unsigned char *host_arr = calloc(resolution[0].height*resolution[0].width*3, sizeof(*host_arr));
    FILE *fp=fopen("nv12.raw","r");
    fread(host_arr,1,NUM_ELEMENTS,fp);
    fclose(fp);
    int i=1000;
   //while(i--)
    {
    	convert(host_arr,host_arr);
    }
	fp=fopen("rgb.tmp","w");
	fwrite(host_arr,1,resolution[0].height*resolution[0].width*3,fp);
	fclose(fp);
	convert_deinit();
}
