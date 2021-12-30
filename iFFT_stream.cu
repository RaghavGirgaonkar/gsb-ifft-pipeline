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

void run_cuFFT(char *arr, int NUM_IFFTS, int NUM_STREAMS, int gpu_id){

    char * buffer = arr;
    //Allocate input data
    printf("Allocating input data\n");
    cufftComplex ***h_in; 
    h_in = new cufftComplex **[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_in[ii] = new cufftComplex *[NUM_IFFTS];
        for (int jj = 0; jj < NUM_IFFTS; ++jj) {
            h_in[ii][jj] = new cufftComplex[NX];
            for (int kk = 0; kk < NX; ++kk) {
                h_in[ii][jj][kk].x = (float)*buffer++;
                h_in[ii][jj][kk].y = (float)*buffer++;
            }
        }
    }
    printf("Allocated input data\n");

    //Allocate output data
    printf("Allocating output data\n");
    cufftComplex ***h_out; 
    h_out = new cufftComplex **[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_out[ii] = new cufftComplex *[NUM_IFFTS];
        for (int jj = 0; jj < NUM_IFFTS; ++jj) {
            h_out[ii][jj] = new cufftComplex[NX];
            for (int kk = 0; kk < NX; ++kk) {
                h_out[ii][jj][kk].x = 0.f;
                h_out[ii][jj][kk].y = 0.f;
            }
        }
    }
    printf("Allocated output data\n");

    // Pin host input and output memory for cudaMemcpyAsync.
    printf("Pinning host input and output moemory for cudaMemcpyAsync\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS; jj++) {
            cudaHostRegister(h_in[ii][jj], NX*sizeof(cufftComplex), cudaHostRegisterPortable);
            cudaHostRegister(h_out[ii][jj], NX*sizeof(cufftComplex), cudaHostRegisterPortable);

        }
    }
    printf("Pinned host input and output moemory for cudaMemcpyAsync\n");

    //Allocate pointers to device input and output arrays
    printf("Allocating pointers to device input and output arrays\n");
    cufftComplex ***d_in = new cufftComplex **[NUM_STREAMS];
    cufftComplex ***d_out = new cufftComplex **[NUM_STREAMS];
    printf("Allocated pointers to device input and output arrays\n");

    // Allocate intput and output arrays on device.
    printf("Allocating input and output arrays on device\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        d_in[ii] = new cufftComplex *[NUM_IFFTS];
        d_out[ii] = new cufftComplex *[NUM_IFFTS];
        for (int jj = 0; jj < NUM_IFFTS; jj++) {
            cudaMalloc((void**)&d_in[ii][jj], NX*sizeof(cufftComplex));
            cudaMalloc((void**)&d_out[ii][jj], NX*sizeof(cufftComplex));
        }
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
    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS*NUM_IFFTS);
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS; jj++) {
            cufftPlan1d(&plans[ii*NUM_IFFTS + jj], NX, CUFFT_C2C, 1);
            cufftSetStream(plans[ii*NUM_IFFTS + jj], streams[ii]);
        }
    }
    printf("Created cuFFT plans and setting them in Streams\n");

    // Fill streams with async memcopies and FFTs.
    printf("Filling Streams\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS; jj++) {

            cudaMemcpyAsync(d_in[ii][jj], h_in[ii][jj], NX*sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[ii]);
            cufftExecC2C(plans[ii*NUM_IFFTS + jj], (cufftComplex*)d_in[ii][jj], (cufftComplex*)d_out[ii][jj], CUFFT_INVERSE);
            cudaMemcpyAsync(h_out[ii][jj], d_out[ii][jj], NX*sizeof(cufftComplex), cudaMemcpyDeviceToHost, streams[ii]);
        }
    }
    printf("Filled Streams\n");

    // Wait for calculations to complete.
    printf("Synchronising Streams\n");
    for(int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaStreamSynchronize(streams[ii]);
    }
    printf("Synchronised Streams\n");
    
    //Normalising Output (to be added)
    printf("Normalising Output\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS; ++jj) {
            for (int kk = 0; kk < 3; ++kk) {
                h_out[ii][jj][kk].x = h_out[ii][jj][kk].x/(float)NX;
                h_out[ii][jj][kk].y = h_out[ii][jj][kk].y/(float)NX;
            }
        }
    }
    printf("Normalised Output\n");

    //Printing a few output values 
    // for (int ii = 0; ii < NUM_STREAMS; ii++) {
    //     printf("Printing for STREAM %d\n", ii);
    //     for (int jj = 0; jj < NUM_IFFTS; ++jj) {
    //         printf("Printing for NUM_IFFT =  %d\n", jj);
    //         for (int kk = 0; kk < 3; ++kk) {
    //             printf("OG : %f %f  Inverse %f %f \n",h_in[ii][jj][kk].x, h_in[ii][jj][kk].y,h_out[ii][jj][kk].x,h_out[ii][jj][kk].y);
    //         }
    //     }
    // }

    // Free memory and streams.
    printf("Freeing memory and streams\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS; jj++) {
        cudaHostUnregister(h_in[ii][jj]);
        cudaHostUnregister(h_out[ii][jj]);
        cudaFree(d_in[ii][jj]);
        cudaFree(d_out[ii][jj]);
        delete[] h_in[ii][jj];
        delete[] h_out[ii][jj];
        
        }
        cudaStreamDestroy(streams[ii]);
    }
    printf("Freed memory and streams\n");

    delete plans;

    cudaDeviceReset(); 

    
}

int main(int argc, const char *argv[])
{
    time_t start, stop;
    char *infile, *buffer, *buffer2, *buffer3, *buffer4; 
	// short int *data1_int;
	// int op_file;
	// const double time_per_block=671088.64; // uSec.
	int nint;
	int data_size;
    const int NUM_IFFTS = 10;
    const int NUM_STREAMS = 339;

	if(argc != 3) {
		printf("Invalid number of parameters! \n");
		printf("Usage: ./%s  <vlt file1> <GPU ID>\n",argv[0]);
		exit(-1);
	}

    int gpu_id = atoi(argv[2]);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
        oops(__LINE__, "cudaSetDevice FAILED\n");

    // int iterations = atoi(argv[2]);

	nint= 3;
	data_size = 2*2048*nint; // 2 for real & Imaginary data.
 
    
    infile = read_file(argv[1]);
    buffer = infile;
    buffer2 = buffer + 30510*2048; 
    buffer3 = buffer2 + 30510*2048;
    buffer4 = buffer3 + 30510*2048; 

    start = time(NULL);

    // thread t1(run_cuFFT, buffer, NUM_IFFTS, NUM_STREAMS, 0);
    // thread t2(run_cuFFT, buffer2, NUM_IFFTS, NUM_STREAMS, 1);
    // thread t3(run_cuFFT, buffer3, NUM_IFFTS, NUM_STREAMS, 2);
    // // run_cuFFT(buffer, NUM_IFFTS, NUM_STREAMS, gpu_id);

    // if (t1.joinable())
    // {
    //     t1.join();
    //     // cout << "Thread with id " << id1 << " is terminated" << endl;
    // }
    // if (t2.joinable())
    // {
    //     t2.join();
    //     // cout << "Thread with id " << id2 << " is terminated" << endl;
    // }
    // if (t3.joinable())
    // {
    //     t3.join();
    //     // cout << "Thread with id " << id3 << " is terminated" << endl;
    // }

    // run_cuFFT(buffer, NUM_IFFTS, NUM_STREAMS, gpu_id);
    // printf("Ran for buffer\n");
    // run_cuFFT(buffer2, NUM_IFFTS, NUM_STREAMS, gpu_id);
    // printf("Ran for buffer2\n");
    // run_cuFFT(buffer3, NUM_IFFTS, NUM_STREAMS, gpu_id);
    // printf("Ran for buffer3\n");
    // run_cuFFT(buffer4, NUM_IFFTS, NUM_STREAMS, gpu_id);
    // printf("Ran for buffer4\n");

    for(int i = 0; i < 36; i++){
       run_cuFFT(buffer, NUM_IFFTS, NUM_STREAMS, gpu_id);
       buffer += NUM_IFFTS*NUM_STREAMS*NX;
       printf("Ran for buffer%d\n", i+1); 
    }

    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);

    return 0;
}