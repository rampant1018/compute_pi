#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M_PI acos(-1.0)

#define SAMPLE_SIZE 10

double compute_pi_baseline(size_t dt)
{
	double pi = 0.0;
	double delta = 1.0 / dt;
	for (size_t i = 0; i < dt; i++) {
		double x = (double) i / dt;
		pi += delta / (1.0 + x * x);
	}
	
	return pi * 4.0;
}

double compute_pi_avx(size_t dt)
{
	double pi = 0.0;
	double delta = 1.0 / dt;
	register __m256d ymm0, ymm1, ymm2, ymm3, ymm4;
	ymm0 = _mm256_set1_pd(1.0);
	ymm1 = _mm256_set1_pd(delta);
	ymm2 = _mm256_set_pd(delta * 3, delta * 2, delta * 1, 0.0);
	ymm4 = _mm256_setzero_pd();

	for (int i = 0; i <= dt - 4; i += 4) {
		ymm3 = _mm256_set1_pd(i * delta);
		ymm3 = _mm256_add_pd(ymm3, ymm2);
		ymm3 = _mm256_mul_pd(ymm3, ymm3);
		ymm3 = _mm256_add_pd(ymm0, ymm3);
		ymm3 = _mm256_div_pd(ymm1, ymm3);
		ymm4 = _mm256_add_pd(ymm4, ymm3);
	}
	double tmp[4] __attribute__((aligned(32)));
	_mm256_store_pd(tmp, ymm4);
	pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];

	return pi * 4.0;
}

double compute_pi_leibniz(size_t n)
{
	double sum = 0.0;
	for (size_t i = 0; i < n; i++) {
		int sign = i % 2 == 0 ? 1 : -1;
		sum += (sign / (2.0 * (double)i + 1.0));
	}

	return sum * 4.0;
}

double compute_pi_leibniz_avx(size_t n)
{
	double pi = 0.0;
	register __m256d ymm0, ymm1, ymm2, ymm3, ymm4;

	ymm0 = _mm256_setzero_pd();
	ymm1 = _mm256_set1_pd(2.0);
	ymm2 = _mm256_set1_pd(1.0);
	ymm3 = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
	
	for (int i = 0; i <= n - 4; i += 4) {
		ymm4 = _mm256_set_pd(i, i + 1.0, i + 2.0, i + 3.0);
		ymm4 = _mm256_mul_pd(ymm4, ymm1);
		ymm4 = _mm256_add_pd(ymm4, ymm2);
		ymm4 = _mm256_div_pd(ymm3, ymm4);
		ymm0 = _mm256_add_pd(ymm0, ymm4);
	}
	double tmp[4] __attribute__((aligned(32)));
	_mm256_store_pd(tmp, ymm0);
	pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];

	return pi * 4.0;
}


double compute_pi_leibniz_fma(size_t n)
{
	double pi = 0.0;
	register __m256d ymm0, ymm1, ymm2, ymm3, ymm4;

	ymm0 = _mm256_setzero_pd();
	ymm1 = _mm256_set1_pd(2.0);
	ymm2 = _mm256_set1_pd(1.0);
	ymm3 = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
	
	for (int i = 0; i <= n - 4; i += 4) {
		ymm4 = _mm256_set_pd(i, i + 1.0, i + 2.0, i + 3.0);
		ymm4 = _mm256_fmadd_pd(ymm1, ymm4, ymm2);
		ymm4 = _mm256_div_pd(ymm3, ymm4);
		ymm0 = _mm256_add_pd(ymm0, ymm4);
	}
	double tmp[4] __attribute__((aligned(32)));
	_mm256_store_pd(tmp, ymm0);
	pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];

	return pi * 4.0;
}

// Calculate 95% confidence interval
// store the interval [min, max] in the first two parameters
// with a set of data which has SAMPLE_SIZE elements
// return mean value
double compute_ci(double *min, double *max, double data[SAMPLE_SIZE])
{
	double mean = 0.0;
	double stddev = 0.0;
	double stderror;

	// Calculate mean value
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		mean += data[i];
	}
	mean /= SAMPLE_SIZE;

	// Calculate standard deviation
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		stddev += pow((data[i] - mean), 2);
	}
	stddev = sqrt(stddev / (double)SAMPLE_SIZE);

	// Calculate standard error
	stderror = stddev / sqrt((double)SAMPLE_SIZE);

	*min = mean - (2.0 * stderror);
	*max = mean + (2.0 * stderror);

	return mean;
}

int main(int argc, char* argv[])
{
	char operation[32];
	size_t n = atoi(argv[2]);
	strncpy(operation, argv[1], 32);

	clock_t begin, end;
	double time_spent[SAMPLE_SIZE];
	double min, max;

	double (*compute_pi)(size_t);
	char method_name[32];
	char time_filename[32];
	char error_filename[32];

	if (!strcmp(operation, "baseline")) {
		compute_pi = &compute_pi_baseline;
		strcpy(method_name, "compute_pi_baseline");
		strcpy(time_filename, "time_baseline.txt");
		strcpy(error_filename, "error_baseline.txt");
	} else if (!strcmp(operation, "avx")) {
		compute_pi = &compute_pi_avx;
		strcpy(method_name, "compute_pi_avx");
		strcpy(time_filename, "time_avx.txt");
		strcpy(error_filename, "error_avx.txt");
	} else if (!strcmp(operation, "leibniz")) {
		compute_pi = &compute_pi_leibniz;
		strcpy(method_name, "compute_pi_leibniz");
		strcpy(time_filename, "time_leibniz.txt");
		strcpy(error_filename, "error_leibniz.txt");
	} else if (!strcmp(operation, "leibniz_avx")) {
		compute_pi = &compute_pi_leibniz_avx;
		strcpy(method_name, "compute_pi_leibniz_avx");
		strcpy(time_filename, "time_leibniz_avx.txt");
		strcpy(error_filename, "error_leibniz_avx.txt");
	} else if (!strcmp(operation, "leibniz_avx_opt")) {
		compute_pi = &compute_pi_leibniz_avx;
		strcpy(method_name, "compute_pi_leibniz_avx_opt");
		strcpy(time_filename, "time_leibniz_avx_opt.txt");
		strcpy(error_filename, "error_leibniz_avx_opt.txt");
	} else if (!strcmp(operation, "leibniz_fma")) {
		compute_pi = &compute_pi_leibniz_fma;
		strcpy(method_name, "compute_pi_leibniz_fma");
		strcpy(time_filename, "time_leibniz_fma.txt");
		strcpy(error_filename, "error_leibniz_fma.txt");
	}

	for (int i = 0; i < SAMPLE_SIZE; i++) {
		begin = clock();
		compute_pi(n * 1000000);
		end = clock();
		time_spent[i] = (double)(end - begin) / CLOCKS_PER_SEC;
	}
	double mean_time = compute_ci(&min, &max, time_spent);

	double pi = compute_pi(n * 1000000);
	double diff = pi - M_PI > 0 ? pi - M_PI : M_PI - pi;
	double error = diff / M_PI;

	printf("%s(%zuM) needs time in 95%% confidence interval[%lf, %lf]\n",
	       method_name, n, min, max);
	printf("error rate = %.15lf\n", error);

	FILE *fw = fopen(time_filename, "a");
	fprintf(fw, "%zu %lf\n", n, mean_time);
	fclose(fw);

	fw = fopen(error_filename, "a");
	fprintf(fw, "%zu %.15lf\n", n, error);
	fclose(fw);

	return 0;
}
