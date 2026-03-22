cp ./phold_ConsSync/SETTINGS.h ./

nvcc -arch=sm_80 -rdc=true -o bench_ConsSync \
	./main_bench.cu \
	./kernels.cu \
	./queues.cu \
	./random.cu \
\
	./phold_ConsSync/model.cu \
	./phold_ConsSync/Event.cu
