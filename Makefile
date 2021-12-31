CC=gcc
CUDA_PATH=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc
LIBDIRS=/usr/local/cuda/lib64/
CUDAFLAGS= -l :libcufft.so -std=c++11

all:
	$(NVCC) -o ifft_stream iFFT_stream.cu read_file.c -L $(LIBDIRS) $(CUDAFLAGS)