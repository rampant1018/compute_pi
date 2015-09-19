#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M_PI acos(-1.0)

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

double compute_pi_leibniz(size_t dt)
{
	double sum = 0.0;
	for (size_t i = 0; i < dt; i++) {
		int sign = i % 2 == 0 ? 1 : -1;
		sum += (sign / (2.0 * (double)i + 1.0));
	}

	return sum * 4.0;
}

int main(int argc, char* argv[])
{
	unsigned int operation = atoi(argv[1]);
	size_t dt = atoi(argv[2]);
	clock_t begin, end;
	double time_spent;
	double (*compute_pi)(size_t);
	char method_name[32];

	switch(operation) {
		case 0:
			compute_pi = &compute_pi_baseline;
			strcpy(method_name, "compute_pi_baseline");
			break;
		case 1:
			compute_pi = &compute_pi_avx;
			strcpy(method_name, "compute_pi_avx");
			break;
		case 2:
			compute_pi = &compute_pi_leibniz;
			strcpy(method_name, "compute_pi_leibniz");
			break;
		default:
			break;
	}

	begin = clock();
	double pi = compute_pi(dt);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	double diff = pi - M_PI > 0 ? pi - M_PI : M_PI - pi;
	double error = diff / M_PI;

	printf("%s(%zu) needs %lf sec.\n", method_name, dt, time_spent);
	printf("error rate = %.15lf\n", error);

	return 0;
}
