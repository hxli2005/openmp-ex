#include <stdio.h>
#include <omp.h>
#include <time.h>

// 获取当前时间（秒）
static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    /* 
     * 实验设计：
     * 我们设计一个极小的工作负载（Small Workload）。
     * 单个循环仅仅加法 10000 次。这点计算量对 CPU 来说可能不到几十微秒就能完成。
     * 我们分别对比【单线程处理它】和【强行拉起分配给多个线程处理它】的时间差，
     * 从而直观展现 OpenMP 创建、分配、同步线程所产生的“额外负担（Overhead）”。
     */
    long small_steps = 10000; 
    volatile double dummy_sum = 0.0; // 加 volatile 防止极简循环被编译器直接给 -O3 干掉了导致测不到时间

    // ---- 1. 预热运行，排除缓存冷启动影响 ----
    for (long i = 0; i < small_steps; ++i) { dummy_sum += i; }

    // ---- 2. 串行（非并行）执行 ----
    dummy_sum = 0.0;
    double start_serial = get_time();
    for (long i = 0; i < small_steps; ++i) {
        dummy_sum += i;
    }
    double time_serial = get_time() - start_serial;

    
    // ---- 3. 并行（OMP）执行 ----
    dummy_sum = 0.0;
    int threads_to_use = 8; // 开8个线程
    omp_set_num_threads(threads_to_use);
    
    double start_parallel = get_time();
    #pragma omp parallel for reduction(+:dummy_sum)
    for (long i = 0; i < small_steps; ++i) {
        dummy_sum += i;
    }
    double time_parallel = get_time() - start_parallel;

    // ---- 打出对比结果 ----
    printf("加法循环 (%ld 次)\n", small_steps);
    printf("串行执行时间: %10.9f 秒\n", time_serial);
    printf("并行执行时间: %10.9f 秒 (使用 %d 线程)\n", time_parallel, threads_to_use);


    return 0;
}
