cp ./phold_RevsComp/SETTINGS.h ./

nvcc -arch=sm_80 -rdc=true -o bench_RevsComp \
	./main_bench.cu \
	./kernels.cu \
	./queues.cu \
	./random.cu \
\
	./phold_RevsComp/model.cu \
	./phold_RevsComp/Event.cu
