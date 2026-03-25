#include <omp.h>//omp_pi_parallel_3.c
#include <stdio.h>
static long num_steps = 50;
double step;
#define NUM_THREADS 2 
void main () 
{	  
	int i; 	  
	double x, sum, pi=0.0; 
	step = 1.0/(double) num_steps; 
	omp_set_num_threads(NUM_THREADS); 
	#pragma omp parallel private (x, sum) 
	{	
		int id = omp_get_thread_num(); 
	  	for (i=id,sum=0.0;i< num_steps;i=i+NUM_THREADS){ 
			printf("thread %d processing i=%d\n", id, i);
		  	x = (i+0.5)*step; 
		  	sum += 4.0/(1.0+x*x); 
	  	} 
		#pragma omp critical
	  		pi += sum ;
	}
	pi = pi * step;
	printf("pi : %.16f\n", pi);
}