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
    long num_steps = 10000000; // 1000万次累加
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    printf("目标：开 4 个线程，将一个共享累加器加到 %ld\n\n", num_steps);

    // ---- 1. 无同步裸跑 (Data Race / 数据竞争) ----
    long counter_naked = 0;
    double start_naked = get_time();
    #pragma omp parallel for
    for (long i = 0; i < num_steps; i++) {
        counter_naked++; // 线程极高频地同时读取、修改、写回这个共享变量
    }
    double time_naked = get_time() - start_naked;
    
    printf("【实验 1：无同步保护】\n");
    printf("期望结果: %ld\n", num_steps);
    printf("实际结果: %ld \n丢了 %ld 次累加\n", counter_naked, num_steps - counter_naked);
    printf("执行耗时: %.4f 秒\n\n", time_naked);


    // ---- 2. 临界区同步 (Critical Section) ----
    long counter_critical = 0;
    double start_critical = get_time();
    #pragma omp parallel for
    for (long i = 0; i < num_steps; i++) {
        // Critical：一次只允许一个线程进入大括号内部。也就是强行变成了排队
        #pragma omp critical
        {
            counter_critical++;
        }
    }
    double time_critical = get_time() - start_critical;
    
    printf("【实验 2：使用临界区强力锁 (#pragma omp critical)】\n");
    printf("实际结果: %ld\n", counter_critical);
    printf("执行耗时: %.4f 秒\n\n", time_critical);


    // ---- 3. 原子操作同步 (Atomic Update) ----
    long counter_atomic = 0;
    double start_atomic = get_time();
    #pragma omp parallel for
    for (long i = 0; i < num_steps; i++) {
        // Atomic：直接利用底层 CPU 指令集架构带的物理原子锁，极速、专一。
        #pragma omp atomic
        counter_atomic++;
    }
    double time_atomic = get_time() - start_atomic;
    
    printf("【实验 3：使用处理器底层原子指令 (#pragma omp atomic)】\n");
    printf("实际结果: %ld \n", counter_atomic);
    printf("执行耗时: %.4f 秒\n\n", time_atomic);


    // ---- 4. 显式时间点同步 (Barrier / 屏障栅栏) ----
    printf("【实验 4：任务的时间点屏障同步 (#pragma omp barrier)】\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        //让偶数号线程拖延 0.2 秒
        if (tid % 2 == 0) {
            double dummy_time = get_time();
            while (get_time() - dummy_time < 0.2); // 忙等阻塞 0.2 秒
            printf("    -[阶段 A] 慢的线程 %d 完成。\n", tid);
        } else {
            printf("    -[阶段 A] 快的线程 %d 完了\n", tid);
        }

        #pragma omp barrier 
        
        #pragma omp single 
        printf("所有线程到Barrier，同时进入下一阶段\n");

        printf("    +[阶段 B] 线程 %d 开启了新的协同任务。\n", tid);
    }
    printf("\n");

    return 0;
}
