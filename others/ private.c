#include <stdio.h>
#include <omp.h>

int main() {
    int shared_var = 100;
    int private_var = 100;

    printf("--- Before Parallel Region ---\n");
    printf("shared_var  = %d\n", shared_var);
    printf("private_var = %d\n\n", private_var);

    // 设置开4个线程
    omp_set_num_threads(4);
    
    /* 
     * 开始并行区域：
     * - shared_var 没有标明，默认就是 shared（所有线程共享同一块内存）
     * - private_var 显式声明为 private（每个线程在此时会在自己的栈里临时抠出一块新内存，名字也叫 private_var）
     */
    #pragma omp parallel private(private_var)
    {
        int tid = omp_get_thread_num();
        
        // 【注意】private 变量进来的时候是没有初始值的！理论上是垃圾值
        // 甚至它并没有继承外面的 '100'
        // 因此每个线程都要针对自己的那一份 private_var 重新赋值
        private_var = tid * 10; 
        
        #pragma omp critical
        {
            shared_var += 1;
            printf("Thread %d: modified private_var = %2d, modified shared_var = %d\n", 
                   tid, private_var, shared_var);
        }
    }

    printf("\n--- After Parallel Region ---\n");
    printf("shared_var  = %d\n", shared_var);
    printf("private_var = %d\n", private_var);

    return 0;
}
