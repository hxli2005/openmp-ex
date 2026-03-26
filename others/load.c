#include <stdio.h>
#include <omp.h>
#include <time.h>

// 获取当前时间（秒）
static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// 模拟一个不均匀的计算任务：传入的参数越大，执行越慢
void heavy_work(int load) {
    volatile long dummy = 0;
    for (int i = 0; i < load * 20000; i++) {
        dummy += i;
    }
}

int main() {
    int num_steps = 400; // 模拟400个任务
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    double thread_time_static[4] = {0};
    double start_static = get_time();

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        double t_start = get_time();

        #pragma omp for schedule(static, 100) nowait
        for (int i = 0; i < num_steps; i++) {
            heavy_work(i);  // i 越大，该任务耗时越久
        }
        
        thread_time_static[tid] = get_time() - t_start;
    }
    double total_static = get_time() - start_static;

    printf("static\n");
    for (int t = 0; t < num_threads; t++) {
        printf("线程 %d 实际工作: %.4f 秒\n", t, thread_time_static[t]);
    }
    printf("--> [静态] 整体总耗时: %.4f 秒 \n\n", total_static);


    // ---- 2. 动态调度 (Dynamic Schedule) ----
    // 工作窃取行为：不预先分好。每个线程做完手头的一块（默认1个），再排队去拿下一个任务
    // 这样不会有人闲着，也不会有人累死
    double thread_time_dynamic[4] = {0};
    double start_dynamic = get_time();

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        double t_start = get_time();
        
        // 使用 dynamic 动态分配
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_steps; i++) {
            heavy_work(i);
        }
        
        thread_time_dynamic[tid] = get_time() - t_start;
    }
    double total_dynamic = get_time() - start_dynamic;

    printf("dynamic\n");
    for (int t = 0; t < num_threads; t++) {
        printf("线程 %d 实际工作耗时: %.4f 秒\n", t, thread_time_dynamic[t]);
    }
    printf("--> [动态] 整体总耗时: %.4f 秒\n\n", total_dynamic);

    if (total_dynamic < total_static) {
        printf("使用动态调度比静态调度快了 %.2f 倍\n", total_static / total_dynamic);
    }
    return 0;
}
