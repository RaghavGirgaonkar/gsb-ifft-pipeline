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
    int n[] = { 2*NX };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = NX+1, odist = 2*NX; // --- Distance between batches for input and output respectively
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = NUM_IFFTS;                  // Number of batches or IFFTS
    float *imag = new float[1];

    imag[0] = 0.f;

    char * buffer = arr;
    //Allocate input data
    printf("Allocating input data\n");
    cufftComplex **h_in; 
    h_in = new cufftComplex *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_in[ii] = new cufftComplex [NUM_IFFTS*(NX+1)];
        for (int jj = 0; jj < NUM_IFFTS; jj++) {
            for (int kk = 0; kk < NX; kk++) {
                h_in[ii][jj*(NX+1) + kk].x = (float)buffer[ii*NUM_IFFTS*2*NX + jj*2*NX + 2*kk];
                h_in[ii][jj*(NX+1) +kk].y = (float)buffer[ii*NUM_IFFTS*2*NX + jj*2*NX + 2*kk + 1];
            }
            //For the last (2049th) complex number, set real part to imaginary part of 0th complex number, set imag to 0 and set imag of 0th complex number to 0
            h_in[ii][jj*(NX+1)+NX].x = h_in[ii][jj*(NX+1)].y;
            h_in[ii][jj*(NX+1)+NX].y = 0.f;
            h_in[ii][jj*(NX+1)].y = 0.f;
        }
    }
    printf("Allocated input data\n");

    //Allocate output data
    printf("Allocating output data\n");
    cufftReal **h_out; 
    h_out = new cufftReal *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_out[ii] = new cufftReal [NUM_IFFTS*2*NX];
        for (int jj = 0; jj < NUM_IFFTS*2*NX; jj++) {
                h_out[ii][jj]= 0.f;
        }
    }
    printf("Allocated output data\n");

    // Pin host input and output memory for cudaMemcpyAsync.
    printf("Pinning host input and output moemory for cudaMemcpyAsync\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
            cudaHostRegister(h_in[ii], NUM_IFFTS*(NX+1)*sizeof(cufftComplex), cudaHostRegisterPortable);
            cudaHostRegister(h_out[ii], NUM_IFFTS*2*NX*sizeof(cufftReal), cudaHostRegisterPortable);

    }
    printf("Pinned host input and output moemory for cudaMemcpyAsync\n");

    //Allocate pointers to device input and output arrays
    printf("Allocating pointers to device input and output arrays\n");
    cufftComplex **d_in = new cufftComplex *[NUM_STREAMS];
    cufftReal **d_out = new cufftReal *[NUM_STREAMS];
    printf("Allocated pointers to device input and output arrays\n");

    // Allocate intput and output arrays on device.
    printf("Allocating input and output arrays on device\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        // d_in[ii] = new cufftComplex [NUM_IFFTS*NX];
        // d_out[ii] = new cufftComplex [NUM_IFFTS*NX];
        
        cudaMalloc((void**)&d_in[ii], NUM_IFFTS*(NX+1)*sizeof(cufftComplex));
        cudaMalloc((void**)&d_out[ii], NUM_IFFTS*2*NX*sizeof(cufftReal));

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
            cufftPlanMany(&plans[ii], rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, batch);
            cufftSetStream(plans[ii], streams[ii]);
    }
    printf("Created cuFFT plans and setting them in Streams\n");

    // Fill streams with async memcopies and FFTs.
    printf("Filling Streams and RUNNING CUFFT_INVERSE\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {

        cudaMemcpyAsync(d_in[ii], h_in[ii], NUM_IFFTS*(NX+1)*sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[ii]);
        cufftExecC2R(plans[ii], (cufftComplex*)d_in[ii], (cufftReal*)d_out[ii]);
        cudaMemcpyAsync(h_out[ii], d_out[ii], NUM_IFFTS*2*NX*sizeof(cufftReal), cudaMemcpyDeviceToHost, streams[ii]);
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
        for (int jj = 0; jj < NUM_IFFTS*2*NX; jj++) {
            h_out[ii][jj] = h_out[ii][jj]/(float)(2*NX);
        }
    }
    printf("Normalised Output\n");

    //Printing a few output values 
    // for (int ii = 0; ii < NUM_STREAMS; ii++) {
    //     printf("Printing for STREAM %d\n", ii);
    //     for (int jj = 0; jj < 3; jj++) {
    //         // printf("Printing for STR =  %d\n", jj);
    //         printf("OG : %f %f  Inverse %f %f \n",h_in[ii][jj].x, h_in[ii][jj].y,h_out[ii][jj], 0.f);
    //             // printf("%s %s \n", (char*) &h_out[ii][jj][kk].x, (char*) &h_out[ii][jj][kk].y);
            
    //     }
        
    // }

    //Writing to File
    printf("Seeking end of file\n");
    out_file.seekg(0, ios_base::end); //Seek end of file to start writing
    printf("Went to end of file\n");
    printf("********Writing to file********\n");
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        for (int jj = 0; jj < NUM_IFFTS*2*NX; jj++) {
                out_file.write((char *)&h_out[ii][jj], sizeof(char));
                out_file.write((char*) &imag[0], sizeof(char)); 
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
    // char *infile, *buffer, *buffer2, *buffer3, *buffer4; 
	// short int *data1_int;
	// int op_file;
	// const double time_per_block=671088.64; // uSec.
    const int NUM_IFFTS = 4069;
    
    const int NUM_STREAMS = 3;

    //NUM_IFFTS*NUM_STREAMS = num_blocks

	if(argc != 4) {
		printf("Invalid number of parameters! \n");
		printf("Usage: ./%s  <input file> <output file> <GPU ID>\n",argv[0]);
		exit(-1);
	}

    int gpu_id = atoi(argv[3]);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
        oops(__LINE__, "cudaSetDevice FAILED\n");

    // int iterations = atoi(argv[2]);
    
    




    
    // infile = read_file(argv[1]);
    
    fstream out_file;
    out_file.open(argv[2], ios::binary | ios::out);
    if(!out_file)
   {
       cout<<"Error in creating file!!!";
       return 0;
   }
  
   cout<<"File created successfully.\n";

    // buffer = infile;
    // buffer2 = buffer + 30510*2048; 
    // buffer3 = buffer2 + 30510*2048;
    // buffer4 = buffer3 + 30510*2048; 

    start = time(NULL);
    //Read file in chunks
    size_t block_size = 4096;
    size_t num_blocks = 12207;
    int file_size;
    double total_num_blocks;

    // char* file_contents = new char [block_size*num_blocks];

    //Open Input File
    ifstream in_file;
    in_file.open(argv[1], ios::binary| ios::in);

    //Get File Size
    in_file.seekg(0,ios::end);
    file_size = in_file.tellg();
    in_file.seekg(0, ios::beg);

    total_num_blocks = file_size/(double)(4096);

    printf("The total number of blocks is %d\n", (int) total_num_blocks);

    //Read data into buffer
    for (int i = 0; i < 10; i++) {
        char* file_contents = new char [block_size*num_blocks];
        in_file.read(file_contents, block_size*num_blocks);

        printf("Running for buffer %d\n", i);
        
        run_cuFFT(file_contents, NUM_IFFTS, NUM_STREAMS, gpu_id, out_file);
        delete[] file_contents;
        printf("Done for buffer %d\n",i);
    }
    out_file.close();
    in_file.close();
    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);
    printf("The total number of blocks is %d\n", (int) total_num_blocks);

    return 0;
}