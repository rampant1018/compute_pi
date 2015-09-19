CFLAGS += -mavx -std=c99 -Wall -Werror -O0 -lm

all: compute_pi

compute_pi: compute_pi.c
	gcc $(CFLAGS) $< -o $@

clean:
	rm compute_pi
