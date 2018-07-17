/*
 * test.c
 *
 *  Created on: 2018Äê6ÔÂ28ÈÕ
 *      Author: admin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <sys/stat.h>
#include <pthread.h>

typedef enum {
  GST_STATE_VOID_PENDING        = 0,
  GST_STATE_NULL                = 1,
  GST_STATE_READY               = 2,
  GST_STATE_PAUSED              = 3,
  GST_STATE_PLAYING             = 4
} GstState;

typedef struct
{
unsigned int offset;
unsigned int width;
unsigned int height;
}picturelayout_t;


#define MAX_RESOLUTION_NUM  16
typedef struct{
picturelayout_t layout[MAX_RESOLUTION_NUM];
char *buffer;
}picture_t;




int gstreamer_start (float scaleratio,int argc, char * argv[]);//block
picture_t *gstreamer_get_picture();//nonblock
int gstreamer_release_picture(picture_t *p);//nonblock
void gstreamer_exit ();//nonblock
int gstreamer_state(GstState *state);//nonblock

int argc=2;
char argv0[]={"gstreamer"};
char argv1[]={"file:///mnt/1.mp4"};
//char argv1[]={"rtsp://192.168.6.155/test.ts"};
char *argv[2]={argv0,argv1};

void gstreamer_thread(void *p)
{
	printf("%s %d\n",__FUNCTION__,__LINE__);
	gstreamer_start(0.7,argc,argv);
	printf("%s %d\n",__FUNCTION__,__LINE__);
}

int main()
{


	char filename[30];
	 pthread_t thread;


		pthread_create(&thread,NULL,gstreamer_thread,NULL);
		GstState state;
		picture_t *p=0;
		unsigned long frame_cnt=0;
		while(frame_cnt>=0){
			gstreamer_state(&state);
			if(state==GST_STATE_PLAYING){

				p=0;
				p=gstreamer_get_picture();
				if(p)
				{
					frame_cnt++;


					for(int i=0;i<MAX_RESOLUTION_NUM;i++){

						//printf("%p %d %u %u %u\n",p,i, p->layout[i].offset, p->layout[i].width,p->layout[i].height);

						if(p->layout[i].height>0 && p->layout[i].width>0){

							sprintf(filename,"%d_%u_%u.gbr",frame_cnt,p->layout[i].width,p->layout[i].height);
							  FILE *fp =fopen(filename,"w");
							   fwrite(p->buffer+p->layout[i].offset,3,p->layout[i].width*p->layout[i].height,fp);
							   fclose(fp);
						}



					}
					// usleep(20*1000);
					//printf("frame cnt %d\n",frame_cnt);


					gstreamer_release_picture(p);
				}
			}else{

				if(frame_cnt>0){
					gstreamer_exit();
					pthread_join(thread,0);
					break;
				}
				usleep(10*1000);

				//printf("state=%d\n",state);

			}
		}

}
