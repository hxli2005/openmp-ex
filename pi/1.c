#include <omp.h>
#include <stdio.h>
static long num_steps = 1000000;
double step; 
#define NUM_THREADS 10
int main () 
{	  int i; 	  
	  double x, pi, sum[NUM_THREADS]; 
	  step = 1.0/(double) num_steps; 
	  omp_set_num_threads(NUM_THREADS); //
	 #pragma omp parallel 
	 {	  
		double x;     
		int id; 
	  	id = omp_get_thread_num(); 
	  	for (i=id, sum[id]=0.0;i< num_steps; i=i+NUM_THREADS){//

		  	x = (i+0.5)*step; 
		  	sum[id] += 4.0/(1.0+x*x); 
	  	} 
	 } 
	  for(i=0, pi=0.0;i<NUM_THREADS;i++)
		pi += sum[i] * step; 

		 printf("pi : %.16f\n", pi);
		return 0;
}
