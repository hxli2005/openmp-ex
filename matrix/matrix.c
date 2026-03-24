#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <Accelerate/Accelerate.h>


static double now_seconds(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void init_random_matrix(double *m, int n) {
	int total = n * n;
	for (int i = 0; i < total; ++i) {
		m[i] = (double)(rand() % 1000) / 100.0;
	}
}

static void zero_matrix(double *m, int n) {
	int total = n * n;
	for (int i = 0; i < total; ++i) {
		m[i] = 0.0;
	}
}

//串行矩阵乘法，输入矩阵a、b，输出矩阵c，规模n*n
static void serial_matmul(const double *a, const double *b, double *c, int n) {
	for (int i = 0; i < n; ++i) {
		int row = i * n;
		for (int k = 0; k < n; ++k) {
            //a[i][k] * b[k][j] -> a[row + k] * b[k * n + j]
			double aik = a[row + k];
			int bk = k * n;
			for (int j = 0; j < n; ++j) {
				c[row + j] += aik * b[bk + j];
			}
		}
	}
}

// 并行方法1：并行外层i循环，static调度
static void parallel_matmul_static(const double *a, const double *b, double *c, int n) {
#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		int row = i * n;
		for (int k = 0; k < n; ++k) {
			double aik = a[row + k];
			int bk = k * n;
			for (int j = 0; j < n; ++j) {
				c[row + j] += aik * b[bk + j];
			}
		}
	}
}

// 并行方法2：先转置b，再并行计算
static void transpose_matrix(const double *src, double *dst, int n) {
#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			dst[j * n + i] = src[i * n + j];
		}
	}
}


static void parallel_matmul_transposed(const double *a, const double *bt, double *c, int n) {
#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		int arow = i * n;
		int crow = i * n;
		for (int j = 0; j < n; ++j) {
			double sum = 0.0;
			int btrow = j * n;
			for (int k = 0; k < n; ++k) {
				sum += a[arow + k] * bt[btrow + k];
			}
			c[crow + j] = sum;
		}
	}
}

//单级切块（使用转置B以优化缓存）
static void parallel_matmul_single_blocked(const double *a, const double *b, double *c, int n, int block_size) {
#pragma omp parallel for
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                int i_max = (ii + block_size > n) ? n : ii + block_size;
                int j_max = (jj + block_size > n) ? n : jj + block_size;
                int k_max = (kk + block_size > n) ? n : kk + block_size;
                for (int i = ii; i < i_max; ++i) {
                    int arow = i * n;
                    int crow = i * n;
                    for (int j = jj; j < j_max; ++j) {
                        double sum = 0.0;
                        int bk = j * n;
                        for (int k = kk; k < k_max; ++k) {
                            sum += a[arow + k] * b[bk + k];
                        }
                        c[crow + j] += sum;
                    }
                }
            }
        }
    }
}
//2级切块
static void parallel_matmul_double_blocked(const double *a, const double *b, double *c, int n, int macro_block_size, int micro_block_size) {
#pragma omp parallel for collapse(2) schedule(static)
	for (int i_macro = 0; i_macro < n; i_macro += macro_block_size) {
		for (int j_macro = 0; j_macro < n; j_macro += macro_block_size) {
			int i_macro_max = (i_macro + macro_block_size > n) ? n : i_macro + macro_block_size;
			int j_macro_max = (j_macro + macro_block_size > n) ? n : j_macro + macro_block_size;

			for (int k_macro = 0; k_macro < n; k_macro += macro_block_size) {
				int k_macro_max = (k_macro + macro_block_size > n) ? n : k_macro + macro_block_size;

				for (int i_micro = i_macro; i_micro < i_macro_max; i_micro += micro_block_size) {
					int i_micro_max = (i_micro + micro_block_size > i_macro_max) ? i_macro_max : i_micro + micro_block_size;

					for (int j_micro = j_macro; j_micro < j_macro_max; j_micro += micro_block_size) {
						int j_micro_max = (j_micro + micro_block_size > j_macro_max) ? j_macro_max : j_micro + micro_block_size;

						for (int i = i_micro; i < i_micro_max; ++i) {
							int arow = i * n;
							int crow = i * n;
							for (int j = j_micro; j < j_micro_max; ++j) {
								double sum = 0.0;
								int brow = j * n;
								for (int k = k_macro; k < k_macro_max; ++k) {
									sum += a[arow + k] * b[brow + k];
								}
								c[crow + j] += sum;
							}
						}
					}
				}
			}
		}
	}
}

// Apple Accelerate BLAS 方法：使用 cblas_dgemm
// C = A * B，利用 Apple 优化的 BLAS 库和多核支持
static void parallel_matmul_accelerate(const double *a, const double *b, double *c, int n) {
	// cblas_dgemm: C = alpha * A * B + beta * C
	// CblasRowMajor: 行优先存储（C 语言惯例）
	// CblasNoTrans: 不转置 A 和 B
	// n, n, n: m=n（A: n×n），n=n（B 的列数），k=n（B: n×n）
	// 1.0, 0.0: alpha=1.0, beta=0.0（只计算 A*B，不加到既有的 C）
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            n, n, n,
	            1.0,
	            a, n,
	            b, n,
	            0.0,
	            c, n);
}

typedef enum {
	METHOD_STATIC = 1,
	METHOD_TRANSPOSED = 2,
	METHOD_SINGLE_BLOCKED = 3,
	METHOD_DOUBLE_BLOCKED = 4,
	METHOD_ACCELERATE = 5
} MethodType;

static const char *method_name(MethodType method) {
	switch (method) {
		case METHOD_STATIC: return "static";
		case METHOD_TRANSPOSED: return "transposed";
		case METHOD_SINGLE_BLOCKED: return "single_blocked";
		case METHOD_DOUBLE_BLOCKED: return "double_blocked";
		case METHOD_ACCELERATE: return "accelerate";
		default: return "unknown";
	}
}

static int parse_method(const char *s, MethodType *method) {
	if (strcmp(s, "static") == 0) {
		*method = METHOD_STATIC;
		return 1;
	}
	if (strcmp(s, "transposed") == 0) {
		*method = METHOD_TRANSPOSED;
		return 1;
	}
	if (strcmp(s, "single") == 0 || strcmp(s, "single_blocked") == 0) {
		*method = METHOD_SINGLE_BLOCKED;
		return 1;
	}
	if (strcmp(s, "double") == 0 || strcmp(s, "double_blocked") == 0) {
		*method = METHOD_DOUBLE_BLOCKED;
		return 1;
	}
	if (strcmp(s, "accelerate") == 0 || strcmp(s, "blas") == 0) {
		*method = METHOD_ACCELERATE;
		return 1;
	}
	return 0;
}


static void get_time_string(char *buf, size_t buf_size) {
	time_t now = time(NULL);
	struct tm *tm_info = localtime(&now);
	strftime(buf, buf_size, "%Y-%m-%d %H:%M:%S", tm_info);
}

static int append_result_csv(
	const char *csv_path,
	const char *method,
	int threads,
	int n,
	double serial_time,
	double parallel_time,
	double speedup
) {
	FILE *fp = fopen(csv_path, "a+");
	if (fp == NULL) {
		return 0;
	}

	fseek(fp, 0, SEEK_END);
	long file_size = ftell(fp);
	if (file_size == 0) {
		fprintf(fp, "timestamp,method,threads,n,serial_time,parallel_time,speedup\n");
	}

	char ts[32];
	get_time_string(ts, sizeof(ts));
	fprintf(fp, "%s,%s,%d,%d,%.6f,%.6f,%.6f\n",
		ts,
		method,
		threads,
		n,
		serial_time,
		parallel_time,
		speedup);

	fclose(fp);
	return 1;
}

int main(int argc, char *argv[]) {
	int n = 10000;
	int threads = 8;
	MethodType method = METHOD_STATIC;
	int block_size = 64;
	int macro_block_size = 128;
	int micro_block_size = 32;
	const char *csv_path = "matrix_results_new.csv";

	if (argc >= 4) {
		n = atoi(argv[1]);
		threads = atoi(argv[2]);
		if (!parse_method(argv[3], &method)) {
			fprintf(stderr, "Invalid method: %s\n", argv[3]);
			fprintf(stderr, "Usage: %s n threads method [block|macro micro] [csv_path]\n", argv[0]);
			return 1;
		}
		if (method == METHOD_SINGLE_BLOCKED) {
			if (argc >= 5) {
				block_size = atoi(argv[4]);
			}
			if (argc >= 6) {
				csv_path = argv[5];
			}
		} else if (method == METHOD_DOUBLE_BLOCKED) {
			if (argc >= 5) {
				macro_block_size = atoi(argv[4]);
			}
			if (argc >= 6) {
				micro_block_size = atoi(argv[5]);
			}
			if (argc >= 7) {
				csv_path = argv[6];
			}
		} else {
			if (argc >= 5) {
				csv_path = argv[4];
			}
		}
	} else {
		printf("Input matrix size N (for NxN): ");
		if (scanf("%d", &n) != 1) {
			fprintf(stderr, "Invalid matrix size input.\n");
			return 1;
		}
		printf("Input thread count: ");
		if (scanf("%d", &threads) != 1) {
			fprintf(stderr, "Invalid thread input.\n");
			return 1;
		}
		printf("Select method (1=static, 2=transposed, 3=single_blocked, 4=double_blocked, 5=accelerate): ");
		int method_input = 1;
		if (scanf("%d", &method_input) != 1 || method_input < 1 || method_input > 5) {
			fprintf(stderr, "Invalid method input.\n");
			return 1;
		}
		method = (MethodType)method_input;
		if (method == METHOD_SINGLE_BLOCKED) {
			printf("Input single block size (e.g. 64): ");
			if (scanf("%d", &block_size) != 1) {
				fprintf(stderr, "Invalid single block size input.\n");
				return 1;
			}
		}
		if (method == METHOD_DOUBLE_BLOCKED) {
			printf("Input macro block size (e.g. 128): ");
			if (scanf("%d", &macro_block_size) != 1) {
				fprintf(stderr, "Invalid macro block size input.\n");
				return 1;
			}
			printf("Input micro block size (e.g. 32): ");
			if (scanf("%d", &micro_block_size) != 1) {
				fprintf(stderr, "Invalid micro block size input.\n");
				return 1;
			}
		}
	}

	if (n <= 0) {
		fprintf(stderr, "Invalid matrix size: %d\n", n);
		fprintf(stderr, "Usage: %s [n] [threads] [csv_path]\n", argv[0]);
		return 1;
	}
	if (threads <= 0) {
		fprintf(stderr, "Invalid threads: %d\n", threads);
		fprintf(stderr, "Usage: %s n threads method [block|macro micro] [csv_path]\n", argv[0]);
		return 1;
	}
	if (block_size <= 0 || macro_block_size <= 0 || micro_block_size <= 0) {
		fprintf(stderr, "Invalid block size.\n");
		return 1;
	}

	size_t bytes = (size_t)n * (size_t)n * sizeof(double);
	double total_mem_gb = (double)(bytes * 6ULL) / (1024.0 * 1024.0 * 1024.0);
	printf("Estimated memory for 6 matrices: %.2f GB\n", total_mem_gb);

	srand((unsigned int)time(NULL));

	printf("Matrix multiplication benchmark\n");
	printf("Size: %dx%d\n", n, n);
	printf("Method: %s\n", method_name(method));
	if (method == METHOD_SINGLE_BLOCKED) {
		printf("Single block size: %d\n", block_size);
	}
	if (method == METHOD_DOUBLE_BLOCKED) {
		printf("Macro block size: %d, Micro block size: %d\n", macro_block_size, micro_block_size);
	}
	printf("Threads: %d\n", threads);
	printf("CSV output: %s\n", csv_path);
	printf("System max OpenMP threads: %d\n\n", omp_get_max_threads());

	omp_set_num_threads(threads);

	double *a = (double *)malloc(bytes);
	double *b = (double *)malloc(bytes);
	double *bt = (double *)malloc(bytes);
	double *c_serial = (double *)malloc(bytes);
	double *c_parallel_static = (double *)malloc(bytes);
	double *c_parallel_transposed = (double *)malloc(bytes);

	if (a == NULL || b == NULL || bt == NULL || c_serial == NULL || c_parallel_static == NULL || c_parallel_transposed == NULL) {
		fprintf(stderr, "Memory allocation failed for N=%d\n", n);
		free(a);
		free(b);
		free(bt);
		free(c_serial);
		free(c_parallel_static);
		free(c_parallel_transposed);
		return 1;
	}

	init_random_matrix(a, n);
	init_random_matrix(b, n);

	zero_matrix(c_serial, n);
	double t0 = now_seconds();
	serial_matmul(a, b, c_serial, n);
	double t1 = now_seconds();
	double ts = t1 - t0;

	double tp = 0.0;
	double speedup = 0.0;

	if (method == METHOD_STATIC) {
		zero_matrix(c_parallel_static, n);
		double t2 = now_seconds();
		parallel_matmul_static(a, b, c_parallel_static, n);
		double t3 = now_seconds();
		tp = t3 - t2;
	}

	if (method == METHOD_TRANSPOSED) {
		double t2 = now_seconds();
		transpose_matrix(b, bt, n);
		zero_matrix(c_parallel_transposed, n);
		parallel_matmul_transposed(a, bt, c_parallel_transposed, n);
		double t3 = now_seconds();
		tp = t3 - t2;
	}

	if (method == METHOD_SINGLE_BLOCKED) {
		double t2 = now_seconds();
		transpose_matrix(b, bt, n);
		double t_transpose = now_seconds();
		zero_matrix(c_parallel_transposed, n);
		parallel_matmul_single_blocked(a, bt, c_parallel_transposed, n, block_size);
		double t3 = now_seconds();
		printf("  [timing breakdown] transpose: %.6f s, blocked_matmul: %.6f s\n", t_transpose - t2, t3 - t_transpose);
		tp = t3 - t2;
	}

	if (method == METHOD_DOUBLE_BLOCKED) {
		double t2 = now_seconds();
		transpose_matrix(b, bt, n);
		zero_matrix(c_parallel_transposed, n);
		parallel_matmul_double_blocked(a, bt, c_parallel_transposed, n, macro_block_size, micro_block_size);
		double t3 = now_seconds();
		tp = t3 - t2;
	}

	if (method == METHOD_ACCELERATE) {
		double t2 = now_seconds();
		zero_matrix(c_parallel_static, n);
		parallel_matmul_accelerate(a, b, c_parallel_static, n);
		double t3 = now_seconds();
		tp = t3 - t2;
	}

	speedup = ts / tp;

	printf("N=%d\n", n);
	printf("  serial         : %.6f s\n", ts);
	printf("  omp %-14s: %.6f s, speedup=%.3f\n", method_name(method), tp, speedup);

	if (!append_result_csv(csv_path, method_name(method), threads, n, ts, tp, speedup)) {
		fprintf(stderr, "warning: failed to append csv for %s N=%d\n", method_name(method), n);
	}

	free(a);
	free(b);
	free(bt);
	free(c_serial);
	free(c_parallel_static);
	free(c_parallel_transposed);

	return 0;
}
