/* Seriel Code */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
	METHOD_SERIAL = 0,
	METHOD_REDUCTION_STATIC,
	METHOD_REDUCTION_DYNAMIC,
	METHOD_ATOMIC,
	METHOD_CRITICAL,
	METHOD_COUNT
} Method;

static const char *method_name(Method m) {
	switch (m) {
	case METHOD_SERIAL:
		return "serial";
	case METHOD_REDUCTION_STATIC:
		return "omp_reduction_static";
	case METHOD_REDUCTION_DYNAMIC:
		return "omp_reduction_dynamic";
	case METHOD_ATOMIC:
		return "omp_atomic";
	case METHOD_CRITICAL:
		return "omp_critical";
	default:
		return "unknown";
	}
}

static double now_seconds(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void get_time_string(char *buf, size_t buf_size) {
	time_t now = time(NULL);
	struct tm *tm_info = localtime(&now);
	strftime(buf, buf_size, "%Y-%m-%d %H:%M:%S", tm_info);
}

static double pi_serial(long num_steps) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	for (long i = 0; i < num_steps; ++i) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	return step * sum;
}

static double pi_omp_reduction_static(long num_steps, int threads) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : sum) schedule(static)
	for (long i = 0; i < num_steps; ++i) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	return step * sum;
}

static double pi_omp_reduction_dynamic(long num_steps, int threads) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : sum) schedule(dynamic, 2048)
	for (long i = 0; i < num_steps; ++i) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	return step * sum;
}

static double pi_omp_atomic(long num_steps, int threads) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
	for (long i = 0; i < num_steps; ++i) {
		double x = (i + 0.5) * step;
		double val = 4.0 / (1.0 + x * x);
#pragma omp atomic
		sum += val;
	}
	return step * sum;
}

static double pi_omp_critical(long num_steps, int threads) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	omp_set_num_threads(threads);
#pragma omp parallel
	{
		double local_sum = 0.0;
#pragma omp for schedule(static)
		for (long i = 0; i < num_steps; ++i) {
			double x = (i + 0.5) * step;
			local_sum += 4.0 / (1.0 + x * x);
		}
#pragma omp critical
		{ sum += local_sum; }
	}
	return step * sum;
}

static double run_method(Method m, long num_steps, int threads) {
	switch (m) {
	case METHOD_SERIAL:
		return pi_serial(num_steps);
	case METHOD_REDUCTION_STATIC:
		return pi_omp_reduction_static(num_steps, threads);
	case METHOD_REDUCTION_DYNAMIC:
		return pi_omp_reduction_dynamic(num_steps, threads);
	case METHOD_ATOMIC:
		return pi_omp_atomic(num_steps, threads);
	case METHOD_CRITICAL:
		return pi_omp_critical(num_steps, threads);
	default:
		return 0.0;
	}
}

static int append_csv(const char *csv_path,
					  const char *method,
					  int threads,
					  long num_steps,
					  int repeats,
					  double time_sec,
					  double speedup,
					  double pi,
					  double abs_error) {
	FILE *fp = fopen(csv_path, "a+");
	if (fp == NULL) {
		return 0;
	}

	fseek(fp, 0, SEEK_END);
	long file_size = ftell(fp);
	if (file_size == 0) {
		fprintf(fp,
				"timestamp,method,threads,num_steps,repeats,time_sec,speedup,pi,abs_error\n");
	}

	char ts[32];
	get_time_string(ts, sizeof(ts));
	fprintf(fp,
			"%s,%s,%d,%ld,%d,%.9f,%.6f,%.12f,%.3e\n",
			ts,
			method,
			threads,
			num_steps,
			repeats,
			time_sec,
			speedup,
			pi,
			abs_error);

	fclose(fp);
	return 1;
}

int main(int argc, char *argv[]) {
	long num_steps = 100000000;
	int repeats = 3;
	int threads = 8;
	const char *csv_path = "pi_results.csv";

	if (argc >= 2) {
		num_steps = atol(argv[1]);
	}
	if (argc >= 3) {
		repeats = atoi(argv[2]);
	}
	if (argc >= 4) {
		threads = atoi(argv[3]);
	}
	if (argc >= 5) {
		csv_path = argv[4];
	}

	if (num_steps <= 0 || repeats <= 0 || threads <= 0) {
		fprintf(stderr, "Usage: %s [num_steps] [repeats] [threads] [csv_path]\n", argv[0]);
		return 1;
	}

	printf("PI benchmark framework\n");
	printf("num_steps: %ld\n", num_steps);
	printf("repeats  : %d\n", repeats);
	printf("threads  : %d\n", threads);
	printf("csv      : %s\n", csv_path);
	printf("max omp threads: %d\n\n", omp_get_max_threads());

	double serial_best = 1e100;
	double serial_pi = 0.0;

	for (int r = 0; r < repeats; ++r) {
		double t0 = now_seconds();
		double pi = run_method(METHOD_SERIAL, num_steps, 1);
		double t1 = now_seconds();
		double elapsed = t1 - t0;
		if (elapsed < serial_best) {
			serial_best = elapsed;
			serial_pi = pi;
		}
	}

	double serial_err = fabs(serial_pi - M_PI);
	printf("%-24s threads=%2d  time=%10.6f s  speedup=%7.3f  pi=%.12f  err=%.3e\n",
		   method_name(METHOD_SERIAL),
		   1,
		   serial_best,
		   1.0,
		   serial_pi,
		   serial_err);
	append_csv(csv_path,
			   method_name(METHOD_SERIAL),
			   1,
			   num_steps,
			   repeats,
			   serial_best,
			   1.0,
			   serial_pi,
			   serial_err);

	Method methods[] = {
		METHOD_REDUCTION_STATIC,
		METHOD_REDUCTION_DYNAMIC,
		METHOD_ATOMIC,
		METHOD_CRITICAL
	};
	int method_count = (int)(sizeof(methods) / sizeof(methods[0]));

	for (int m = 0; m < method_count; ++m) {
		Method method = methods[m];
		double best_time = 1e100;
		double best_pi = 0.0;

		for (int r = 0; r < repeats; ++r) {
			double t0 = now_seconds();
			double pi = run_method(method, num_steps, threads);
			double t1 = now_seconds();
			double elapsed = t1 - t0;
			if (elapsed < best_time) {
				best_time = elapsed;
				best_pi = pi;
			}
		}

		double speedup = serial_best / best_time;
		double err = fabs(best_pi - M_PI);

		printf("%-24s threads=%2d  time=%10.6f s  speedup=%7.3f  pi=%.12f  err=%.3e\n",
			   method_name(method),
			   threads,
			   best_time,
			   speedup,
			   best_pi,
			   err);

		if (!append_csv(csv_path,
						method_name(method),
						threads,
						num_steps,
						repeats,
						best_time,
						speedup,
						best_pi,
						err)) {
			fprintf(stderr, "warning: failed to append csv\n");
		}
	}

	return 0;
}
