#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <fftw3.h>
#include <stdint.h>

#define LINUX
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define NCHAN 2048


void run_fftw( fftw_plan *p, int8_t *in_stream, fftw_complex *in, double *out, int nchan){
    int i;
    // printf("Instream:\n");
    // for (i=0; i<3 ;i++)
    // {
    //     printf("Input: %f %f\n", (double)in_stream[2*i], (double)in_stream[2*i+1]);
    // }
    double norm = 1.0/sqrt(2*nchan);
    for (i=0; i<nchan; i++)
    {   in[i][0] = norm * (double)(in_stream[2*i+0]);
        in[i][1] = norm * (double)(in_stream[2*i+1]);
    }
    in[nchan][0] = (double)(in[0][1]); //real (n) = imag(0); copied
    in[nchan][1] = norm * (double)0.0;     // imag(n) = 0
    in[0][1] =     norm * (double)0.0; //real (n) = imag(0); copied

    // printf("In\n");
    // for (i=0; i<3 ;i++)
    // {
    //     printf("Input: %f %f\n", in[i][0], in[i][1]);
    // }

    fftw_execute(*p);

    //Print some values
    // printf("New Block\n");
    // for (i=0; i<3 ;i++)
    // {
    //     printf("Output: %d\n", (int8_t)out[i]);
    // }
}

long int fillbuf(double *inbuf, int8_t *buf_to_fill, int NX, long int lastptr)
{
  long int ptrnow;
  int i;
  int8_t imag = 0;
  for (i=0; i<NX; i++){
        *(buf_to_fill+lastptr+2*i) = (int8_t)inbuf[i];
        *(buf_to_fill+lastptr+2*i+1) = imag;
  }     
  ptrnow = lastptr + 2*NX;

  return ptrnow;
}

void new_fillbuf(double *inbuf, int8_t *buf_to_fill, int NX){
   int i;
  int8_t imag = 0;
  for (i=0; i<NX; i++){
        *(buf_to_fill+i) = (int8_t)inbuf[i];
        // *(buf_to_fill+2*i+1) = imag;
  } 
}

int main(int argc, char* argv[]){

    time_t start, stop;

    if(argc < 5)
    { fprintf(stderr, "USAGE: %s <Input File> <Output File> <Bandwidth of Observation in MHz> <Num Seconds>\n", argv[0]);
      return(-1);
    }

    int NX = 2*NCHAN;
    int nchan = NCHAN;
    FILE *in_file, *out_file;
    uint8_t *in_stream, *out_stream, *framebuf;
    long int read_file;
    int num_blocks;
    int num_seconds;
    int bandwidth;
    long double beam_sampling_rate;

    //Pointers for FFTW in and out
    fftw_complex *in;
    double *out;
    fftw_plan p;

    //Allocate Memory
    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (nchan + 1));
    out = (double*)fftw_malloc(sizeof(double) * NX);

    //Open input and output files
    in_file = fopen(argv[1], "r");
    if(!in_file){
        fprintf(stderr,"Error opening input file\n");
        fftw_free(in);
        fftw_free(out);
        exit(1);
    }
    out_file = fopen(argv[2], "w");
    if(!out_file){
        fprintf(stderr,"Error opening output file\n");
        fftw_free(in);
        fftw_free(out);
        exit(1);
    }

    //Get number of seconds to process
    num_seconds = atoi(argv[4]);
    printf("Number of Seconds to process = %d\n", num_seconds);

    //Get Bandwidth in MHz
    bandwidth = atoi(argv[3]);
    printf("Bandwidth = %d\n", bandwidth);

    //Calculate Beam Sampling rate
    long double sampling_rate = (long double)(1/((long double) 2*bandwidth));
    sampling_rate *= (long double) 0.000001;
    // printf("Sampling rate is = %Lf\n", (long double) sampling_rate);
    beam_sampling_rate = (long double) sampling_rate*4096;
    // printf("Beam Sampling rate is = %Lf\n", beam_sampling_rate);

    //Calculate Number of blocks from Number of Seconds
    num_blocks = (int)(num_seconds/beam_sampling_rate);
    printf("Number of Blocks to process = %d\n", num_blocks);

    printf("Size of output file will be %llu bytes or %d Megabytes\n", (unsigned long int)(2*num_blocks*4096), (int) ((2*num_blocks*4096)/(1000000)));


    //Make FFTW C2R plan
    p = fftw_plan_dft_c2r_1d(NX, in, out, FFTW_MEASURE);

    in_stream = (int8_t*)calloc(2*nchan, sizeof(int8_t));
    out_stream = (int8_t*)calloc(2*nchan, sizeof(int8_t));

    long int lastptr, ptrnow;

    lastptr = 0;
    ptrnow = 0;
    int i = 0;
    start = time(NULL);
    while(i< num_blocks){

        // printf("Running for block number %d out of %d\n", i+1, num_blocks);
        //Running IFFT and filling output buffer
        read_file = fread(in_stream, sizeof(int8_t), 2*nchan, in_file);
        run_fftw(&p, in_stream, in, out, nchan);
        lastptr = ptrnow;
        // ptrnow = fillbuf(out, out_stream, NX, lastptr);
        new_fillbuf(out, out_stream, NX);
        // printf("*****Writing to File for block %d*****\n",i);
        fwrite(out_stream, sizeof(int8_t), 2*nchan, out_file);

        i++;

    }

    printf("Done!\n");
    fftw_free(in);
    fftw_free(out);
    free(in_stream);
    free(out_stream);
    //Writing to file
    // printf("*****Writing to File*****\n");
    // fwrite(out_stream, sizeof(int8_t), 2*2*nchan*num_blocks, out_file);
    // printf("*****Done Writing to File*****\n");
    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);

    return 0;

}