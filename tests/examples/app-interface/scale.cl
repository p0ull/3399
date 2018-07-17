#pragma OPENCL EXTENSION cl_arm_printf : enable

#define MAX_RESOLUTION_NUM  16
typedef struct{
unsigned int width;
unsigned int height;
unsigned int total_width;
unsigned int total_height;
}def_resolution;

const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void scale(__write_only image2d_t destinationImage, __read_only image2d_t sourceImage, __global def_resolution *resolution)
{
			unsigned int i =get_global_id(0);
			unsigned int j =get_global_id(1);
			
			   int2 coordinate = (int2)(get_global_id(0), get_global_id(1));
	
			unsigned int srcwidth =resolution[0].width;
			 unsigned int srcheight =resolution[0].height;
			 
			 

			 unsigned int destwidth =resolution[0].total_width;
			 unsigned int destheight =resolution[0].total_height;

			 //printf("srcwidth=%d srcheight=%d\n",srcwidth,srcheight);
			 //printf("destwidth=%u destheight=%u\n",destwidth,destheight);
			 
			 int index=0;
			 int whit=0,hhit=0;;

					for(index=1;index<MAX_RESOLUTION_NUM;index++){
						  if(i>=resolution[index].total_width && i<(resolution[index].total_width+resolution[index].width)){
						  			whit=index;
						  }
						  						  			
						  if(j>=resolution[index].total_height && j<(resolution[index].total_height+resolution[index].height)){
						  			hhit=index;
						  }
						  
						  if(whit && hhit)
						      break;
						  
					}
			
				
				float2 normalizedCoordinate;
			if(whit && hhit){

				normalizedCoordinate=convert_float2((int2)(i-resolution[whit].total_width, j-resolution[whit].total_height)) * (float2)(1.0f/resolution[whit].width, 1.0f/resolution[whit].height);
									
			}else{
			
				return;
			}
			
		
	//if(whit>1 && hhit>1)
	// printf("x=%d y=%d .x=%f .y=%f\n",coordinate.x,coordinate.y,normalizedCoordinate.x,normalizedCoordinate.y);
			float4 colour = read_imagef(sourceImage, sampler, normalizedCoordinate);
					 printf("x=%d y=%d colour.x=%f colour.y=%f colour.z=%f colour.w=%f \n",coordinate.x,coordinate.y,colour.x,colour.y,colour.z,colour.w);
  		write_imagef(destinationImage, coordinate, colour);
}