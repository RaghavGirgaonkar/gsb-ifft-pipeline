#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <cufft.h>
#include <cufftXt.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <thread>

extern "C" {
    #include "read_file.h"
}

using namespace std;

#define _FILE_OFFSET_BITS 64
#define NX 2048
#define BATCH 1


void oops(int linenum, const char * msg) {
	fprintf(stderr, "\n*** OOPS, fatal error detected at line %d !!\n*** %s !!\n\n", linenum, msg);
	exit(86);
}

void run_test(char *arr, int num_iter, int deviceID){
    printf("\n------------------------------------------------ run_test: Starting on GPU ID \t %d\n",deviceID);
    if(cudaSetDevice(deviceID) != cudaSuccess)
        oops(__LINE__, "cudaSetDevice FAILED\n");
    char * arrptr = arr;
    // Allocate host memory for the signal
    int malloc_size = sizeof(cufftComplex) * NX;
    for(unsigned int j = 0; j< num_iter; j++){
    printf("Using malloc size for h_signal%d\n", malloc_size);
    cufftComplex* h_signal = (cufftComplex*)malloc(malloc_size);
    if (h_signal == NULL)
        oops(__LINE__, "malloc 1 FAILED\n");
    // cufftComplex* h_signal2 = (cufftComplex*)malloc(malloc_size);
    // if (h_signal2 == NULL)
    //     oops(__LINE__, "malloc 2 FAILED\n");

    // Initalize the memory for the signal
    
    // char * arrptr2 = arr + 4096;
    for (unsigned int i = 0; i < NX; i++) {
        h_signal[i].x = (float)*arrptr++;
        h_signal[i].y = (float)*arrptr++;
        // h_signal2[i].x = (float)*arrptr2++;
        // h_signal2[i].y = (float)*arrptr2++;
    }
    printf("Done populating h_signals\n");
    
    // Allocate device memory for signal
    size_t dev_mem_size = sizeof(cufftComplex) * NX;
    printf("Device memory allocation size for each signal = %ld\n", dev_mem_size);
    cufftComplex* d_signal;
    // cufftComplex* d_signal2;
    if(cudaMalloc((void**)&d_signal, dev_mem_size) != cudaSuccess)
       oops(__LINE__, "cudaMalloc d_signal FAILED\n");
    printf("d_signal allocated\n");
    // if(cudaMalloc((void**)&d_signal2, dev_mem_size) != cudaSuccess)
    //    oops(__LINE__, "cudaMalloc d_signal2 FAILED\n");
    // printf("d_signal2 allocated\n");

    // Copy host memory to device
    if(cudaMemcpy(d_signal, h_signal, dev_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
       oops(__LINE__, "cudaMemcpy to d_signal FAILED\n");
    printf("Done copying from Host to Device pt 1\n");
    // if(cudaMemcpy(d_signal2, h_signal2, dev_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
    //    oops(__LINE__, "cudaMemcpy to d_signal2 FAILED\n");
    // printf("Done copying from Host to Device pt 2\n");

    //Create Plan
    cufftHandle plan;
    if(cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
        oops(__LINE__, "cufftPlan1d FAILED\n");
   
    printf("Made plan\n");

    //Execute Inverse Transform
    if (cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE) != CUFFT_SUCCESS)
	    oops(__LINE__, "cufftExecC2C d_signal FAILED");
    // if (cufftExecC2C(plan, (cufftComplex *)d_signal2, (cufftComplex *)d_signal2, CUFFT_INVERSE) != CUFFT_SUCCESS)
	//     oops(__LINE__, "cufftExecC2C d_signal2 FAILED");
    printf("Done with iFFT\n");

    // Synchronize device.
    if(cudaDeviceSynchronize() != cudaSuccess)
	    oops(__LINE__, "cudaDeviceSynchronize FAILED");
    
    //Copy device memory to Host
    cufftComplex* h_inverse_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * NX);
    if(h_inverse_signal == NULL)
        oops(__LINE__, "malloc h_inverse_signal FAILED");
    // cufftComplex* h_inverse_signal2 = (cufftComplex*)malloc(sizeof(cufftComplex) * NX);
    // if(h_inverse_signal2 == NULL)
    //     oops(__LINE__, "malloc h_inverse_signal2 FAILED");
    if(cudaMemcpy(h_inverse_signal, d_signal, dev_mem_size,cudaMemcpyDeviceToHost) != cudaSuccess)
        oops(__LINE__, "cudaMemcpy h_inverse_signal FAILED");
    // if(cudaMemcpy(h_inverse_signal2, d_signal2, dev_mem_size,cudaMemcpyDeviceToHost) != cudaSuccess)
    //     oops(__LINE__, "cudaMemcpy h_inverse_signal2 FAILED");
    printf("Done with copying back to Host\n");

    //Display Inverse Transform
    printf("Printing iFFT for first array:\n");
    for(int i=0;i< 3;i++){
        h_inverse_signal[i].x= h_inverse_signal[i].x/(float)NX;
        h_inverse_signal[i].y= h_inverse_signal[i].y/(float)NX;
        printf("OG : %f %f  Inverse %f %f \n",h_signal[i].x,h_signal[i].y,h_inverse_signal[i].x,h_inverse_signal[i].y);
    }
    // printf("Printing iFFT for second array:\n");
    // for(int i = 0; i < 3; i++){
    //     h_inverse_signal2[i].x= h_inverse_signal2[i].x/(float)NX;
    //     h_inverse_signal2[i].y= h_inverse_signal2[i].y/(float)NX;
    //     printf("OG : %f %f  Inverse %f %f \n",h_signal2[i].x,h_signal2[i].y,h_inverse_signal2[i].x,h_inverse_signal2[i].y);
    // }
    cufftDestroy(plan);
    printf("Done destroying plan\n");
    free(h_signal);
    // free(h_signal2);
    printf("Done freeing hsignals\n");
    free(h_inverse_signal);
    // free(h_inverse_signal2);
    printf("Done freeing h_inverse signals\n");
    cudaFree(d_signal);
    // cudaFree(d_signal2);
    printf("Done freeing d_signals\n");
    cudaDeviceReset();
    printf("Done cudaDeviceReset()\n");
    }

}

int main(int argc, const char *argv[])
{
    time_t start, stop;
    char *infile, *buffer, *buffer2, *buffer3; 
	// short int *data1_int;
	// int op_file;
	// const double time_per_block=671088.64; // uSec.
	int nint;
	int data_size;
    const int NUM_IFFTS = 10;
    const int NUM_STREAMS = 5120;

	if(argc != 4) {
		printf("Invalid number of parameters! \n");
		printf("Usage: ./%s  <vlt file1> <num_iterations> <GPU ID>\n",argv[0]);
		exit(-1);
	}

    int gpu_id = atoi(argv[3]);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
        oops(__LINE__, "cudaSetDevice FAILED\n");

    int iterations = atoi(argv[2]);

	nint= 3;
	data_size = 2*2048*nint; // 2 for real & Imaginary data.
 
    
    infile = read_file(argv[1]);
    buffer = infile;
    buffer2 = buffer + 2*2048; 
    buffer3 = buffer2 + 2*2048; 

    start = time(NULL);

    //Allocate input data
    printf("Allocating input data\n");
    cufftComplex **h_in = new cufftComplex *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_in[ii] = new cufftComplex[NX];
        for (int jj = 0; jj < NUM_IFFTS; ++jj) {
            h_in[ii][jj].x = (float)*buffer++;
            h_in[ii][jj].y = (float)*buffer++;
        }
    }
    printf("Allocated input data\n");

    //Allocate output data
    printf("Allocating output data\n");
    cufftComplex **h_out = new cufftComplex *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_out[ii] = new cufftComplex[NX];
        for (int jj = 0; jj < NX; ++jj) {
            h_out[ii][jj].x = 0.f;
            h_out[ii][jj].y = 0.f;
        }
    }
    printf("Allocated output data\n");

    // Pin host input and output memory for cudaMemcpyAsync.
    printf("Pinning host input and output moemory for cudaMemcpyAsync\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaHostRegister(h_in[ii], NX*sizeof(cufftComplex), cudaHostRegisterPortable);
        cudaHostRegister(h_out[ii], NX*sizeof(cufftComplex), cudaHostRegisterPortable);
    }
    printf("Pinned host input and output moemory for cudaMemcpyAsync\n");

    //Allocate pointers to device input and output arrays
    printf("Allocating pointers to device input and output arrays\n");
    cufftComplex **d_in = new cufftComplex *[NUM_STREAMS];
    cufftComplex **d_out = new cufftComplex *[NUM_STREAMS];
    printf("Allocated pointers to device input and output arrays\n");

    // Allocate intput and output arrays on device.
    printf("Allocating input and output arrays on device\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaMalloc((void**)&d_in[ii], NX*sizeof(cufftComplex));
        cudaMalloc((void**)&d_out[ii], NX*sizeof(cufftComplex));
    }
    printf("Allocated input and output arrays on device\n");

    // Create CUDA streams.
    printf("Creating CUDA Streams\n");
    cudaStream_t streams[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaStreamCreate(&streams[ii]);
    }
    printf("Created CUDA streams\n");

    // Creates cuFFT plans and sets them in streams
    printf("Creating cuFFT plans and setting them in Streams\n");
    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cufftPlan1d(&plans[ii], NX, CUFFT_C2C, 1);
        cufftSetStream(plans[ii], streams[ii]);
    }
    printf("Created cuFFT plans and setting them in Streams\n");

    // Fill streams with async memcopies and FFTs.
    printf("Filling Streams\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        int jj = ii;
        cudaMemcpyAsync(d_in[jj], h_in[jj], NX*sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[jj]);
        cufftExecC2C(plans[jj], (cufftComplex*)d_in[jj], (cufftComplex*)d_out[jj], CUFFT_INVERSE);
        cudaMemcpyAsync(h_out[jj], d_out[jj], NX*sizeof(cufftComplex), cudaMemcpyDeviceToHost, streams[jj]);
    }
    printf("Filled Streams\n");

    // Wait for calculations to complete.
    printf("Synchronising Streams\n");
    for(int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaStreamSynchronize(streams[ii]);
    }
    printf("Synchronised Streams\n");
    
    //Normalising Output (to be added)

    //Printing a few output values 
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        printf("Printing for STREAM %d\n", ii);
        for (int jj = 0; jj < 3; ++jj) {
            printf("OG : %f %f  Inverse %f %f \n",h_in[ii][jj].x, h_in[ii][jj].y,h_out[ii][jj].x,h_out[ii][jj].y);
        }
    }

    // Free memory and streams.
    printf("Freeing memory and streams\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaHostUnregister(h_in[ii]);
        cudaHostUnregister(h_out[ii]);
        cudaFree(d_in[ii]);
        cudaFree(d_out[ii]);
        delete[] h_in[ii];
        delete[] h_out[ii];
        cudaStreamDestroy(streams[ii]);
    }
    printf("Freed memory and streams\n");

    delete plans;

    cudaDeviceReset(); 


    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);

    return 0;
}