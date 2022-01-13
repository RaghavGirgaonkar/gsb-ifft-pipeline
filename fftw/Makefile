CC=gcc
CUDA_PATH=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc
LIBDIRS=/usr/local/cuda/lib64/ 
CUDAFLAGS= -l :libcufft.so -std=c++11
CFLAGS= -O3 -lm -L/usr/local/lib -lfftw3
File=ifft_fftw_new

all:
	$(CC) -o $(File) $(File).c $(CFLAGS)

