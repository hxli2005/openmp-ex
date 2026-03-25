#include <omp.h>
#include <stdio.h>
static long num_steps = 30;
double step; 
#define NUM_THREADS 2
int main () 
{	  int i;//在parallel外部声明，导致循环 	  
	  double x, pi, sum[NUM_THREADS]; 
	  step = 1.0/(double) num_steps; 
	  omp_set_num_threads(NUM_THREADS); //
	 #pragma omp parallel 
	 {	  
		double x; //重复声明    
		int id; 
	  	id = omp_get_thread_num(); 
	  	for (i=id, sum[id]=0.0;i< num_steps; i=i+NUM_THREADS){//
			printf("thread %d processing i=%d\n", id, i);
		  	x = (i+0.5)*step; 
		  	sum[id] += 4.0/(1.0+x*x); 
	  	} 
	 } 
	  for(i=0, pi=0.0;i<NUM_THREADS;i++)
		pi += sum[i] * step; 

		 printf("pi : %.16f\n", pi);
		return 0;
}
