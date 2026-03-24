#include <omp.h>
#include <stdio.h>

int main(void) {
    printf("=== 1) Serial ===\n");
    printf("Hello World from serial program.\n\n");

    printf("=== 2) Parallel (default threads) ===\n");
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Hello World from thread %d of %d (default).\n", tid, nthreads);
    }
    printf("\n");

    printf("=== 3) Parallel (set 8 threads in code) ===\n");
    omp_set_num_threads(8);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Hello World from thread %d of %d (set in code).\n", tid, nthreads);
    }

    return 0;
}