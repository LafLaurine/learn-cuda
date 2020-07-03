CUDA_PATH     ?= /usr/
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCC_DBG       = -g -G
NVCCFLAGS      = $(NVCC_DBG) -m64
# I use a GTX950M and my compute probability is 5.0
GENCODE_FLAGS  = -gencode arch=compute_50,code=sm_50

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg