CFLAGS += -mavx -std=c99 -Wall -Werror -O0

all: compute_pi

compute_pi: compute_pi.c
	gcc $(CFLAGS) $< -o $@ -lm

clean:
	rm compute_pi
