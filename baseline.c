#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char* argv[])
{
	size_t dt = atoi(argv[1]);
	printf("pi = %lf\n", compute_pi_baseline(dt));

	return 0;
}
