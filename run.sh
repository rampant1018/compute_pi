#!/bin/bash
rm time_baseline.txt
rm time_baseline_avx.txt
rm time_leibniz.txt
rm time_leibniz_avx.txt
rm time_leibniz_avx_opt.txt
rm time_leibniz_avx_opt_single.txt
rm time_recursion.txt
rm time_leibniz_JM.txt
rm error_baseline.txt
rm error_baseline_avx.txt
rm error_leibniz.txt
rm error_leibniz_avx.txt
rm error_leibniz_avx_opt.txt
rm error_leibniz_avx_opt_single.txt
rm error_recursion.txt
rm error_leibniz_JM.txt
N=(1 2 4 8 16 32 64 128 256 512 1024)
Operation=(baseline baseline_avx leibniz leibniz_avx leibniz_avx_opt leibniz_avx_opt_single recursion leibniz_JM)
#Operation=(leibniz_JM)
for op in ${Operation[@]}
do
	for n in ${N[@]}
	do
		sde64 -hsw -- ./compute_pi $op $n
	done
done
