/* Seriel Code */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
	METHOD_SERIAL = 0,
	METHOD_1,
	METHOD_2,
	METHOD_3,
	METHOD_4,
	METHOD_COUNT
} Method;

static const char *method_name(Method m) {
	switch (m) {
	case METHOD_SERIAL:
		return "serial";
	case METHOD_1:
		return "1";
	case METHOD_2:
		return "2";
	case METHOD_3:
		return "3";
	case METHOD_4:
		return "4";
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

static double pi_omp_1(long num_steps, int threads) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	double *partial = (double *)calloc((size_t)threads, sizeof(double));
	if (partial == NULL) {
		return 0.0;
	}
	omp_set_num_threads(threads);

#pragma omp parallel for schedule(static, 1)
	for (long i = 0; i < num_steps; ++i) {
		int id = omp_get_thread_num();
		double x = (i + 0.5) * step;
		partial[id] += 4.0 / (1.0 + x * x);
	}

	for (int t = 0; t < threads; ++t) {
		sum += partial[t];
	}

	free(partial);
	return step * sum;
}

static double pi_omp_2(long num_steps, int threads) {
	double step = 1.0 / (double)num_steps;
	double sum = 0.0;
	double *partial = (double *)calloc((size_t)threads, sizeof(double));
	if (partial == NULL) {
		return 0.0;
	}
	omp_set_num_threads(threads);

#pragma omp parallel
	{
		int id = omp_get_thread_num();
		partial[id] = 0.0;

#pragma omp for 
		for (long i = 0; i < num_steps; ++i) {
			double x = (i + 0.5) * step;
			partial[id] += 4.0 / (1.0 + x * x);
		}
	}

	for (int t = 0; t < threads; ++t) {
		sum += partial[t];
	}

	free(partial);
	return step * sum;
}

static double pi_omp_3(long num_steps, int threads) {
	int i; 	  
	double x, sum = 0.0, pi = 0.0; 
	double step = 1.0/(double) num_steps; 
	omp_set_num_threads(threads); 
	#pragma omp parallel private (i,x, sum) 
	{	
		int id = omp_get_thread_num(); 
	  	for (i=id,sum=0.0;i< num_steps;i=i+threads){ 
		  	x = (i+0.5)*step; 
		  	sum += 4.0/(1.0+x*x); 
	  	} 
		#pragma omp critical
	  		pi += sum ;
	}
	return step * pi;
}

static double pi_omp_4(long num_steps, int threads) {
	int i;
	double step = 1.0 / (double)num_steps;
	double x, sum = 0.0; 
	step = 1.0/(double) num_steps; 
	omp_set_num_threads(threads); 
	#pragma omp parallel for reduction(+:sum) private(x) 
	for (i=0;i<num_steps; i++){ 
		 x = (i+0.5)*step; 
		sum = sum + 4.0/(1.0+x*x); 
	} 
	return step * sum;
}

static double run_method(Method m, long num_steps, int threads) {
	switch (m) {
	case METHOD_SERIAL:
		return pi_serial(num_steps);
	case METHOD_1:
		return pi_omp_1(num_steps, threads);
	case METHOD_2:
		return pi_omp_2(num_steps, threads);
	case METHOD_3:
		return pi_omp_3(num_steps, threads);
	case METHOD_4:
		return pi_omp_4(num_steps, threads);
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
	int threads = 16;
	const char *csv_path = "pi_results.csv";
	const double pi_ref = acos(-1.0);

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

	double serial_err = fabs(serial_pi - pi_ref);
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
		METHOD_1,
		METHOD_2,
		METHOD_3,
		METHOD_4
	};
	int method_count = (int)(sizeof(methods) / sizeof(methods[0]));

	for (int m = 0; m < method_count; ++m) {
		Method method = methods[m];
		const char *display_name = method_name(method);
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
		double err = fabs(best_pi - pi_ref);

		printf("%-24s threads=%2d  time=%10.6f s  speedup=%7.3f  pi=%.12f  err=%.3e\n",
			   display_name,
			   threads,
			   best_time,
			   speedup,
			   best_pi,
			   err);

		if (!append_csv(csv_path,
						display_name,
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
