CFLAGS += -mavx -O3 -std=c99

all: baseline avx

baseline: baseline.c
	gcc $(CFLAGS) $< -o $@

avx: avx.c
	gcc $(CFLAGS) $< -o $@

clean:
	rm avx baseline
