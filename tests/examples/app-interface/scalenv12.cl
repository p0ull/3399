#pragma OPENCL EXTENSION cl_arm_printf : enable

#define UOFFSET(width,height,i,j) width*height+((j)>>1)*width+(i)-(i)%2
#define VOFFSET(width,height,i,j) UOFFSET(width,height,i,j)+1
#define YOFFSET(width,height,i,j) (j)*width+(i)

__kernel void scale(__global uchar * dest, __global uchar * src,int srcwidth, int srcheight)
{
			unsigned int i =get_global_id(0);
			unsigned int j =get_global_id(1);
			 unsigned int destwidth =get_global_size(0);
			 unsigned int destheight =get_global_size(1);
			 
			// printf("srcwidth=%d srcheight=%d\n",srcwidth,srcheight);
			// printf("destwidth=%u destheight=%u\n",destwidth,destheight);
			float w_scale_rate = (float)srcwidth / destwidth;
			float h_scale_rate = (float)srcheight / destheight;
			
			//printf("w_scale_rate=%f h_scale_rate=%f\n",w_scale_rate,h_scale_rate);
			float i_scale = w_scale_rate * i;
			float j_scale = h_scale_rate* j;
			
			//printf("i_scale=%f j_scale=%f\n",i_scale,j_scale);
			int srci = i_scale;
			int srcj = j_scale;
			float u = i_scale - srci;
			float v = j_scale - srcj;
			//printf("srci=%d srcj=%d\n",srci,srcj);
			if((srci+1)>srcwidth || (srcj+1)>srcheight){
			//	printf("\nsrci+1=%d srcj+1=%d i_scale=%f j_scale=%f i=%u j=%u w_scale_rate=%f h_scale_rate=%f ttt=%f fff=%f\n",srci+1,srcj+1,i_scale,j_scale,i,j,w_scale_rate,h_scale_rate,h_scale_rate * i,w_scale_rate * j);
			}
			dest[YOFFSET(destwidth,destheight,i,j)]=((1-u)*(1-v)*src[YOFFSET(srcwidth,srcheight,srci,srcj)]+(1-u)*v*src[YOFFSET(srcwidth,srcheight,srci,srcj+1)]+u*(1-v)*src[YOFFSET(srcwidth,srcheight,srci+1,srcj)]+u*v*src[YOFFSET(srcwidth,srcheight,srci+1,srcj+1)]);
			dest[UOFFSET(destwidth,destheight,i,j)]=((1-u)*(1-v)*src[UOFFSET(srcwidth,srcheight,srci,srcj)]+(1-u)*v*src[UOFFSET(srcwidth,srcheight,srci,srcj+1)]+u*(1-v)*src[UOFFSET(srcwidth,srcheight,srci+1,srcj)]+u*v*src[UOFFSET(srcwidth,srcheight,srci+1,srcj+1)]);
			dest[VOFFSET(destwidth,destheight,i,j)]=((1-u)*(1-v)*src[VOFFSET(srcwidth,srcheight,srci,srcj)]+(1-u)*v*src[VOFFSET(srcwidth,srcheight,srci,srcj+1)]+u*(1-v)*src[VOFFSET(srcwidth,srcheight,srci+1,srcj)]+u*v*src[VOFFSET(srcwidth,srcheight,srci+1,srcj+1)]);
}