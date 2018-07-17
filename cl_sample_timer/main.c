//0e_multidim.c
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define NUM_ELEMENTS_X 16
#define NUM_ELEMENTS_Y 16
#define NUM_ELEMENTS (NUM_ELEMENTS_X*NUM_ELEMENTS_Y)

/* Try setting this to 1 to verify that get_global_id()
 * is equivalent to a formula written using get_group_id(),
 * get_local_size() and get_local_id().
 */
#define VERBOSE_GLOBAL_ID 0

/* A kernel which sets all elements of an array to the 1D ID of the work group */
//const char * source_str  = "__kernel void setidx(__global uchar *out)"
//                           "{"
//
//                           /* Get the 2D X and Y indices */
//#if VERBOSE_GLOBAL_ID
//                           "    int index_x = get_group_id(0) * get_local_size(0) + get_local_id(0);"
//                           "    int index_y = get_group_id(1) * get_local_size(1) + get_local_id(1);"
//#else
//                           "    int index_x = get_global_id(0);"
//                           "    int index_y = get_global_id(1);"
//#endif
//
//                           /* Map the two 2D indices to a single linear, 1D index */
//                           "    int grid_width = get_num_groups(0) * get_local_size(0);"
//                           "    int index = index_y * grid_width + index_x;"
//
//                           /* Map the two 2D group indices to a single linear, 1D group index */
//                           "    int result = get_group_id(1) * get_num_groups(0) + get_group_id(0);"
//
//                           /* Write out the result */
//                           "    out[index] = result;"
//                           "}";

const char * source_str  = "__kernel void setidx(__global uchar * rgb)"
						"{"
						" unsigned int i =get_global_id(0);"
						" unsigned int j =get_global_id(1);"
						" unsigned int W =get_global_size(0);"
						" unsigned int H =get_global_size(1);"
						" uchar y,u,v,r,g,b;"
						" y=yuv[j*W+i];"
						" u=yuv[W*H+(i+W*j)/4];"
						" v=yuv[W*H+(i+W*j)/4+1];"
						" r=y+1.403*(v-128);"
						" g=y-0.343*(u-128)-0.714*(v-128);"
						" b=y+1.770*(u-128);"
						" rgb[j*3+i]=1;"
						" rgb[j*3+i+1]=1;"
						" rgb[j*3+i+2]=1;"
						"}";

void printArray(char *host_array, int num_elements){
    size_t i;
    for (i = 0; i < num_elements; ++i)
    {
        printf("%3d ", host_array[i]);
        if (i % 16 == 15)
            printf("\n");
    }
    printf("\n");
}

int main(void) {


	cl_int status;
    /* Get platform and device information */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint num_devices;
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    /* Create an OpenCL context */
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create a command queue */
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Allocate host array. Note that the elements are zeroed. */
    char *host_arr = calloc(NUM_ELEMENTS, sizeof(*host_arr));

    /* Print the zeroed array. */
    printArray(host_arr, NUM_ELEMENTS);

    /* Create device memory buffer */
    cl_mem dev_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        NUM_ELEMENTS * sizeof(*host_arr), NULL, &ret);

    /* Create a program from the kernel source */
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, NULL, &ret);

    /* Build the program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

 	if(ret != 0)
 		{
 		printf("clBuild failed:%d\n", ret);
 		char tbuf[0x10000];
 		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
 		printf("\n%s\n", tbuf);
 		return -1;
 		}
    /* Create the OpenCL kernel */
    cl_kernel kernel = clCreateKernel(program, "setidx", &ret);

    /* Set the kernel argument */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_mem_obj);

    /* Execute the OpenCL kernel */
    size_t global_item_size[2] = { NUM_ELEMENTS_X, NUM_ELEMENTS_Y};
    size_t local_item_size[2] = {4, 4};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
            global_item_size, local_item_size, 0, NULL, NULL);

    /* Read the results from the device memory buffer back into host array */
    ret = clEnqueueReadBuffer(command_queue, dev_mem_obj, CL_TRUE, 0,
                              NUM_ELEMENTS * sizeof(*host_arr), host_arr, 0, NULL, NULL);

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);

    if (ret != CL_SUCCESS) {
        printf("OpenCL error executing kernel: %d\n", ret);
        goto cleanup;
    }

    /* Print the fetched array */
    printArray(host_arr, NUM_ELEMENTS);

cleanup:
    /* Clean up */
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(dev_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}
