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

void run_cuFFT(char *arr, int NUM_IFFTS, int NUM_STREAMS, int gpu_id, fstream& out_file){

    //1D Batched IFFTS
    int rank = 1;                           // --- 1D FFTs
    int n[] = { NX };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = NX, odist = NX; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = NUM_IFFTS;                  // Number of batches or IFFTS

    char * buffer = arr;
    //Allocate input data
    printf("Allocating input data\n");
    cufftComplex **h_in; 
    h_in = new cufftComplex *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_in[ii] = new cufftComplex [NUM_IFFTS*NX];
        for (int jj = 0; jj < NUM_IFFTS*NX; jj++) {
                h_in[ii][jj].x = (float)*buffer++;
                h_in[ii][jj].y = (float)*buffer++;
            
        }
    }
    printf("Allocated input data\n");

    //Allocate output data
    printf("Allocating output data\n");
    cufftComplex **h_out; 
    h_out = new cufftComplex *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_out[ii] = new cufftComplex [NUM_IFFTS*NX];
        for (int jj = 0; jj < NUM_IFFTS*NX; jj++) {
                h_out[ii][jj].x = 0.f;
                h_out[ii][jj].y = 0.f;
        }
    }
    printf("Allocated output data\n");

    // Pin host input and output memory for cudaMemcpyAsync.
    printf("Pinning host input and output moemory for cudaMemcpyAsync\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
            cudaHostRegister(h_in[ii], NUM_IFFTS*NX*sizeof(cufftComplex), cudaHostRegisterPortable);
            cudaHostRegister(h_out[ii], NUM_IFFTS*NX*sizeof(cufftComplex), cudaHostRegisterPortable);

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
        // d_in[ii] = new cufftComplex [NUM_IFFTS*NX];
        // d_out[ii] = new cufftComplex [NUM_IFFTS*NX];
        
        cudaMalloc((void**)&d_in[ii], NUM_IFFTS*NX*sizeof(cufftComplex));
        cudaMalloc((void**)&d_out[ii], NUM_IFFTS*NX*sizeof(cufftComplex));

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
            cufftPlanMany(&plans[ii], rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
            cufftSetStream(plans[ii], streams[ii]);
    }
    printf("Created cuFFT plans and setting them in Streams\n");

    // Fill streams with async memcopies and FFTs.
    printf("Filling Streams and RUNNING CUFFT_INVERSE\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {

        cudaMemcpyAsync(d_in[ii], h_in[ii], NUM_IFFTS*NX*sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[ii]);
        cufftExecC2C(plans[ii], (cufftComplex*)d_in[ii], (cufftComplex*)d_out[ii], CUFFT_INVERSE);
        cudaMemcpyAsync(h_out[ii], d_out[ii], NUM_IFFTS*NX*sizeof(cufftComplex), cudaMemcpyDeviceToHost, streams[ii]);
    }
    printf("Filled Streams AND RAN CUFFT_INVERSE \n");

    // Wait for calculations to complete.
    printf("Synchronising Streams\n");
    for(int ii = 0; ii < NUM_STREAMS; ii++) {
        cudaStreamSynchronize(streams[ii]);
    }
    printf("Synchronised Streams\n");
    
    //Normalising Output (to be added)
    printf("Normalising Output\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS*NX; jj++) {
            h_out[ii][jj].x = h_out[ii][jj].x/(float)NX;
            h_out[ii][jj].y = h_out[ii][jj].y/(float)NX;
        }
    }
    printf("Normalised Output\n");

    //Printing a few output values 
    // for (int ii = 0; ii < NUM_STREAMS; ii++) {
    //     printf("Printing for STREAM %d\n", ii);
    //     for (int jj = 0; jj < 3; jj++) {
    //         // printf("Printing for STR =  %d\n", jj);
    //         printf("OG : %f %f  Inverse %f %f \n",h_in[ii][jj].x, h_in[ii][jj].y,h_out[ii][jj].x,h_out[ii][jj].y);
    //             // printf("%s %s \n", (char*) &h_out[ii][jj][kk].x, (char*) &h_out[ii][jj][kk].y);
            
    //     }
    // }

    //Writing to File
    printf("Seeking end of file\n");
    out_file.seekg(0, ios_base::end); //Seek end of file to start writing
    printf("Went to end of file\n");
    printf("********Writing to file********\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS*NX; jj++) {
                out_file.write((char*)&h_out[ii][jj].x, sizeof(char));
                out_file.write((char*)&h_out[ii][jj].y, sizeof(char)); 
        }
    }
    printf("********Done Writing to file!********\n");

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

    
}

int main(int argc, const char *argv[])
{
    time_t start, stop;
    char *infile, *buffer, *buffer2, *buffer3, *buffer4; 
	// short int *data1_int;
	// int op_file;
	// const double time_per_block=671088.64; // uSec.
    const int NUM_IFFTS = 30;
    const int NUM_STREAMS = 113;

	if(argc != 4) {
		printf("Invalid number of parameters! \n");
		printf("Usage: ./%s  <input file> <output file> <GPU ID>\n",argv[0]);
		exit(-1);
	}

    int gpu_id = atoi(argv[3]);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
        oops(__LINE__, "cudaSetDevice FAILED\n");

    // int iterations = atoi(argv[2]);
 
    
    infile = read_file(argv[1]);
    
    fstream out_file;
    out_file.open(argv[2], ios::binary | ios::out);
    if(!out_file)
   {
       cout<<"Error in creating file!!!";
       return 0;
   }
  
   cout<<"File created successfully.\n";

    buffer = infile;
    buffer2 = buffer + 30510*2048; 
    buffer3 = buffer2 + 30510*2048;
    buffer4 = buffer3 + 30510*2048; 

    start = time(NULL);

    

    for(int i = 0; i < 36; i++){
       run_cuFFT(buffer, NUM_IFFTS, NUM_STREAMS, gpu_id, out_file);
       buffer += NUM_IFFTS*NUM_STREAMS*NX;
       printf("Ran for buffer%d\n", i+1); 
    }

    out_file.close();

    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);

    return 0;
}