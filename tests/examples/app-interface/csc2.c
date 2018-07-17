//0e_multidim.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define NUM_ELEMENTS_X 1920
#define NUM_ELEMENTS_Y 1120
#define NUM_ELEMENTS (NUM_ELEMENTS_X*NUM_ELEMENTS_Y*3)

/* Try setting this to 1 to verify that get_global_id()
 * is equivalent to a formula written using get_group_id(),
 * get_local_size() and get_local_id().
 */
#define VERBOSE_GLOBAL_ID 0

/* A kernel which sets all elements of an array to the 1D ID of the work group */
const char * source_str  =	"__kernel void yuv2rgb(__global uchar * rgb, __global uchar * yuv)"
							"{"
							" unsigned int i =get_global_id(0);"
							" unsigned int j =get_global_id(1);"
							" unsigned int W =get_global_size(0);"
							" unsigned int H =get_global_size(1);"
							" uchar y,u,v,r,g,b;"
							" y=yuv[j*W+i];"
							"unsigned int tmp1=W*H+(j>>1)*W+i-i%2;"
							" u=yuv[tmp1];"
							" v=yuv[tmp1+1];"
							" r=y+1.403f*(v-128);"
							" g=y-0.343f*(u-128)-0.714f*(v-128);"
							" b=y+1.770f*(u-128);"
							" unsigned int tmp3=(j*W+i)*3;"
							" rgb[tmp3]=r;"
							" rgb[tmp3+1]=g;"
							" rgb[tmp3+2]=b;"
							"}";


void printArray(char *host_array, int num_elements){
    size_t i;
    for (i = 0; i < num_elements; ++i)
    {
        printf("%02x", host_array[i]);
        if (i % 16 == 15)
            printf("\n");
    }
    printf("\n");
}

#define QUEUE_SIZE  5

struct queue_element{
	cl_kernel kernel;
	cl_mem gpu_rgb;
	cl_mem gpu_yuv;
	char * gpu_cpu_rgb;
	char *gpu_cpu_yuv;
	cl_event event;
	struct timeval timestart;
	struct timeval timeend;
};

static struct queue_element  eventqueue[QUEUE_SIZE];
static int head = 0;
static int tail = 0;


int enqueue_prepare()
{
    if ((tail + 1) % QUEUE_SIZE == head) {
        printf("queque full \n");
        return -1;
    }else{
    	return tail;
    }
}

void enqueue_do(){
    tail = (tail + 1) % QUEUE_SIZE;
}


int dequeue(){

    int temp;
    if (tail == head) {
        printf("队列为空，元素无法出队列\n");
        return -1;
    }else{

    	temp = head;
        head = (head + 1) % QUEUE_SIZE;
        return temp;
    }

}

//判断队列是否为空
int isempty(){
    if (head == tail) {
        printf("队列为空\n");
        return 1;
    }

    printf("队列不为空\n");
    return 0;
}

int isfull(){

    if ((tail + 1) % QUEUE_SIZE == head) {
        printf("队列已满\n");
        return 1;
    }

    printf("队列未满\n");
    return 0;
}

//打印出队列元素
void PrintQueue(){

    for (int i = head; i < tail; i++) {
        printf("%d ",queue[i]);
    }
    printf("\n");
}


cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_uint num_devices;
cl_uint num_platforms;
cl_int ret;
cl_context context;
cl_command_queue command_queue;
cl_program program;

int csc_init(void) {



    /* Get platform and device information */

    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    /* Create an OpenCL context */
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);


    /* Create a command queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Allocate host array. Note that the elements are zeroed. */

    /* Print the zeroed array. */
   // printArray(host_arr, NUM_ELEMENTS);

    /* Create device memory buffer */

    /* Create a program from the kernel source */
    program = clCreateProgramWithSource(context, 1, &source_str, NULL, &ret);

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




 	for(int i=0;i<QUEUE_SIZE;i++){
		/* Create the OpenCL kernel */
 		eventqueue[i].kernel = clCreateKernel(program, "yuv2rgb", &ret);


 		eventqueue[i].gpu_rgb = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
											NUM_ELEMENTS * sizeof(char), NULL, &ret);
 		eventqueue[i].gpu_yuv = clCreateBuffer(context, CL_MEM_READ_ONLY,
											NUM_ELEMENTS * sizeof(char), NULL, &ret);

		/* Set the kernel argument */
		ret = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &eventqueue[i].gpu_rgb);
		ret = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &eventqueue[i].gpu_yuv);
		eventqueue[i].event = clCreateUserEvent(context, NULL);
 	}



}


#define USEMMAP
int csc_nv12gbr(int width,int height,char *cpu_yuv,char *cpu_rgb)
{

	int index=enqueue_prepare();
	if(index>=0 && index<QUEUE_SIZE){

	}else{
      return -1;
	}
	struct queue_element *pq=&eventqueue[index];

    gettimeofday(&pq->timestart,NULL);

#ifdef USEMMAP

    gpu_cpu_yuv=clEnqueueMapBuffer(command_queue,gpu_yuv,CL_TRUE,CL_MAP_WRITE,0,NUM_ELEMENTS * sizeof(char),0,0,0,0);
    //printf("\ngpu_cpu_yuv=%p\n",gpu_cpu_yuv);


    memcpy(gpu_cpu_yuv,cpu_yuv, width*height*3/2);
    //ret = clFlush(command_queue);
    //ret = clFinish(command_queue);

    clEnqueueUnmapMemObject(command_queue,gpu_yuv,gpu_cpu_yuv,0,0,0);
#else
    ret = clEnqueueWriteBuffer(command_queue, gpu_yuv, CL_TRUE, 0, width*height*3/2 * sizeof(char), cpu_yuv, 0, NULL, NULL);
#endif

   // gettimeofday(&end,NULL);
   // printf("\ntime %ld ms\n",end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000));

   // gettimeofday(&start,NULL);
    /* Execute the OpenCL kernel */
    size_t global_item_size[2] = { width, height};
    size_t local_item_size[2] = {8, 8};
    ret = clEnqueueNDRangeKernel(command_queue, kernel[0], 2, NULL, global_item_size, local_item_size, 0, NULL, &event);


//    ret = clFlush(command_queue);
//    ret = clFinish(command_queue);
    clWaitForEvents(1, &event);

//    gettimeofday(&end,NULL);
//
//    printf("time %ld ms\n",end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000));
//
//
//
//
//
//
//    gettimeofday(&start,NULL);
    /* Read the results from the device memory buffer back into host array */

#ifdef USEMMAP
    gpu_cpu_rgb=clEnqueueMapBuffer(command_queue,gpu_rgb,CL_TRUE,CL_MAP_READ,0,NUM_ELEMENTS * sizeof(char),0,0,0,0);
   // printf("gpu_cpu_rgb=%p\n",gpu_cpu_rgb);

    memcpy(cpu_rgb,gpu_cpu_rgb, width*height*3);
   // ret = clFlush(command_queue);
   // ret = clFinish(command_queue);

    clEnqueueUnmapMemObject(command_queue,gpu_rgb,gpu_cpu_rgb,0,0,0);
#else

     ret = clEnqueueReadBuffer(command_queue, gpu_rgb, CL_TRUE, 0, width*height*3 * sizeof(char), cpu_rgb, 0, NULL, NULL);
#endif
    gettimeofday(&end,NULL);
    long int delta=end.tv_sec*1000+end.tv_usec/1000-(start.tv_sec*1000+start.tv_usec/1000);
    if(delta>30)
    printf("time %ld ms\n",delta);




    if (ret != CL_SUCCESS) {
        printf("OpenCL error executing kernel: %d\n", ret);
       // goto cleanup;
    }






    /* Print the fetched array */
   // printArray(host_arr, NUM_ELEMENTS);



}

int csc_deinit()
{

    ret = clReleaseKernel(kernel[0]);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(gpu_rgb);
    ret = clReleaseMemObject(gpu_yuv);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}


//int main()
//{
//
//	csc_init();
//
//    /* Allocate host array. Note that the elements are zeroed. */
//    unsigned char *host_arr = calloc(NUM_ELEMENTS, sizeof(*host_arr));
//    FILE *fp=fopen("nv12","r");
//    fread(host_arr,1,NUM_ELEMENTS,fp);
//    fclose(fp);
//    while(1){
//    	csc_nv12gbr(1920,1088,host_arr,host_arr);
//    }
//	fp=fopen("rgb","w");
//	fwrite(host_arr,1,NUM_ELEMENTS,fp);
//	fclose(fp);
//	csc_deinit();
//}
