cp ./phold_OptmSync/SETTINGS.h ./

nvcc -arch=sm_80 -rdc=true -o bench_OptmSync \
	./main_bench.cu \
	./kernels.cu \
	./queues.cu \
	./random.cu \
\
	./phold_OptmSync/model.cu \
	./phold_OptmSync/Event.cu
