#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
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

void run_cuFFT(char *arr, int NUM_IFFTS, int gpu_id, FILE* out_file){

    //1D Batched IFFTS
    int rank = 1;                           // --- 1D FFTs
    int n[] = { 2*NX };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = NX+1, odist = 2*NX; // --- Distance between batches for input and output respectively
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = NUM_IFFTS;                  // Number of batches or IFFTS
    int8_t *imag = new int8_t[1];

    imag[0] = 0;

    char * buffer = arr;
    //Allocate input data
    // printf("Allocating input data\n");
    cufftComplex *h_in; 
    h_in = new cufftComplex [NUM_IFFTS*(NX+1)];
    for (int jj = 0; jj < NUM_IFFTS; jj++) {
        for (int kk = 0; kk < NX; kk++) {
            h_in[jj*(NX+1) + kk].x = (float)buffer[jj*2*NX + 2*kk];
            h_in[jj*(NX+1) +kk].y = (float)buffer[jj*2*NX + 2*kk + 1];
        }
        //For the last (2049th) complex number, set real part to imaginary part of 0th complex number, set imag to 0 and set imag of 0th complex number to 0
        h_in[jj*(NX+1)+NX].x = h_in[jj*(NX+1)].y;
        h_in[jj*(NX+1)+NX].y = 0.f;
        h_in[jj*(NX+1)].y = 0.f;
    }
    
    // printf("Allocated input data\n");

    //Allocate output data
    // printf("Allocating output data\n");
    cufftReal *h_out;
    int8_t *h_final; 
    h_out = new cufftReal [NUM_IFFTS*2*NX];
    h_final = new int8_t [NUM_IFFTS*4*NX];
    // printf("Allocated output data\n");

    //Allocate pointers to device input and output arrays
    // printf("Allocating pointers to device input and output arrays\n");
    cufftComplex *d_in = new cufftComplex [NUM_IFFTS*(NX+1)];
    cufftReal *d_out = new cufftReal [NUM_IFFTS*2*NX];
    // printf("Allocated pointers to device input and output arrays\n");

    // Allocate intput and output arrays on device.
    // printf("Allocating input and output arrays on device\n");        
    cudaMalloc((void**)&d_in, NUM_IFFTS*(NX+1)*sizeof(cufftComplex));
    cudaMalloc((void**)&d_out, NUM_IFFTS*2*NX*sizeof(cufftReal));
    // printf("Allocated input and output arrays on device\n");

    // Creates cuFFT plans and sets them in streams
    // printf("Creating cuFFT plan\n");
    cufftHandle plan;
    cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, batch);
    // printf("Created cuFFT plan\n");

    // Run iFFTs.
    // printf("RUNNING CUFFT_INVERSE\n");
    cudaMemcpy(d_in, h_in, NUM_IFFTS*(NX+1)*sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cufftExecC2R(plan, (cufftComplex*)d_in, (cufftReal*)d_out);
    cudaMemcpy(h_out, d_out, NUM_IFFTS*2*NX*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    // printf("RAN CUFFT_INVERSE \n");
    
    //Normalising Output (to be added)
    // printf("Normalising Output\n");
    for (int jj = 0; jj < NUM_IFFTS*2*NX; jj++) {
        h_out[jj] /= (float)sqrt(2*NX);
    }
    // printf("Normalised Output\n");\

    //Making final Output array
    // printf("Making final Output Array\n");
    for (int jj = 0; jj < NUM_IFFTS*2*NX; jj++) {
        h_final[2*jj] = (int8_t)h_out[jj];
        h_final[2*jj+1] = imag[0];
    }

    //Writing to File
    int retval;
    // printf("Went to end of file\n");
    // printf("********Writing to file********\n");
    retval = fwrite(h_final, NUM_IFFTS*4*NX*sizeof(int8_t),1, out_file);    
    // printf("********Done Writing to file!********\n");

    // Free memory and streams.
    // printf("Freeing memory\n");
    cudaFree(d_in);
    cudaFree(d_out);
    delete h_in;
    delete h_out;
    delete h_final;
    cufftDestroy(plan);
    
    // printf("Freed memory\n");

    
    //Reset Device
    cudaDeviceReset(); 

    
}

int main(int argc, const char *argv[])
{
    time_t start, stop;
    long int num_seconds, num_blocks;
    int bandwidth;
    long double beam_sampling_rate;
    size_t block_size = 2*NX;
    size_t blocks_to_send = 100000;


	if(argc != 7) {
		printf("Usage: %s  <Pol1 input file> <Pol2 input file> <output file> <Bandwidth in MHz> <NSec> <GPU ID>\n",argv[0]);
		exit(-1);
	}

    //Get NSecs from user
    num_seconds = atoi(argv[5]);
    printf("Number of Seconds to process = %ld\n", num_seconds);

    //Get Bandwidth
    bandwidth = atoi(argv[4]);
    printf("Bandwidth = %d\n", bandwidth);

    //Calculate Beam Sampling Rate
    long double sampling_rate = (long double)(1/((long double) 2*bandwidth));
    sampling_rate *= (long double) 0.000001;
    beam_sampling_rate = (long double) sampling_rate*4096;

    //Calculate Number of blocks from Number of Seconds
    num_blocks = (long int)(num_seconds/beam_sampling_rate);
    printf("Number of Blocks to process = %ld\n", num_blocks);

    printf("Size of output file will be %ld bytes or %ld Megabytes\n", (long int)(2*2*num_blocks*4096), (long int) ((2*2*num_blocks*4096)/(1000000)));

    //Set GPU 
    int gpu_id = atoi(argv[6]);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
        oops(__LINE__, "cudaSetDevice FAILED\n");

    
    //Opening Output file
    FILE* out_file;
    out_file = fopen(argv[3], "w");
    if(!out_file)
   {
       cout<<"Error in creating file!!!";
       exit(1);
   }
  
   cout<<" Output File created successfully.\n"; 

    start = time(NULL);

    //Open Input Files
    ifstream in_file_pol1, in_file_pol2;
    in_file_pol1.open(argv[1], ios::binary| ios::in);
    if(!in_file_pol1)
   {
       cout<<"Error in opening Pol1 Input file!!!";
       exit(1);
   }
    in_file_pol2.open(argv[2], ios::binary| ios::in);
    if(!in_file_pol2)
   {
       cout<<"Error in opening Pol2 Input file!!!";
       exit(1);
   }

    //Read File in Chunks (<blocks_to_send> number of blocks each)
    int blocks_left = num_blocks;
    while(blocks_left > 0){
        if(blocks_left < blocks_to_send){
            printf("Blocks left %d\n", blocks_left);
                char* file_contents = new char [block_size*blocks_left];

                //Pol1 
                in_file_pol1.read(file_contents, block_size*blocks_left);
                run_cuFFT(file_contents, blocks_left, gpu_id, out_file);

                //Pol2
                in_file_pol2.read(file_contents, block_size*blocks_left);
                run_cuFFT(file_contents, blocks_left, gpu_id, out_file);


                delete[] file_contents;
                blocks_left -= blocks_left;

        }
        else{

            printf("Blocks left %d\n", blocks_left);

            char* file_contents = new char [block_size*blocks_to_send];

            //Pol1
            in_file_pol1.read(file_contents, block_size*blocks_to_send);
            run_cuFFT(file_contents, blocks_to_send, gpu_id, out_file);

            //Pol2
            in_file_pol2.read(file_contents, block_size*blocks_to_send);
            run_cuFFT(file_contents, blocks_to_send, gpu_id, out_file);

            delete[] file_contents;
            blocks_left -= blocks_to_send;
        }


    }

    int close;

    close = fclose(out_file);
    in_file_pol1.close();
    in_file_pol2.close();
    stop = time(NULL);
    printf("Done!\n");
    printf("The number of seconds for to run was %ld\n", stop - start);
    

    return 0;
}