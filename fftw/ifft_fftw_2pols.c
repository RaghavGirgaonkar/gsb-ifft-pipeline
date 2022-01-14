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

void progressbar(int n, long int N)
{
   int i;
   char *tmp;
   int barWidth = 50;
   float progress = (float)(n+1)/(float)N;
   int pos = barWidth * progress;
   fprintf(stdout, "%c",'[');
   for (i = 0; i < barWidth; ++i)
   { if (i < pos)  fprintf(stdout, "%c", '=');
     else if (i == pos) fprintf(stdout, "%c", '>'); 
     else fprintf(stdout, "%c", ' ');
   }
   fprintf(stdout, "] %3d %%\r", (int)(progress * 100.0));
   fflush(stdout);
}


void run_fftw( fftw_plan *p, int8_t *in_stream, fftw_complex *in, double *out, int nchan){
    int i;
    double norm = 1.0/sqrt(2*nchan);
    for (i=0; i<nchan; i++)
    {   in[i][0] = norm * (double)(in_stream[2*i+0]);
        in[i][1] = norm * (double)(in_stream[2*i+1]);
    }
    in[nchan][0] = (double)(in[0][1]); //real (n) = imag(0); copied
    in[nchan][1] = norm * (double)0.0;     // imag(n) = 0
    in[0][1] =     norm * (double)0.0; //real (n) = imag(0); copied

    fftw_execute(*p);
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
        *(buf_to_fill+2*i) = (int8_t)inbuf[i];
        *(buf_to_fill+2*i+1) = imag;
  } 
}

int main(int argc, char* argv[]){

    time_t start, stop;

    if(argc < 6)
    { fprintf(stderr, "USAGE: %s <Input File Pol1> <Input File Pol2> <Output File> <Bandwidth of Observation in MHz> <Num Seconds>\n", argv[0]);
      return(-1);
    }

    int NX = 2*NCHAN;
    int nchan = NCHAN;
    FILE *in_file_pol1, *in_file_pol2, *out_file;
    uint8_t *in_stream_pol1, *out_stream_pol1, *in_stream_pol2, *out_stream_pol2;
    long int read_file;
    long int num_blocks;
    long int num_seconds;
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
    in_file_pol1 = fopen(argv[1], "r");
    if(!in_file_pol1){
        fprintf(stderr,"Error opening Pol1 Input File\n");
        fftw_free(in);
        fftw_free(out);
        exit(1);
    }
    in_file_pol2 = fopen(argv[2], "r");
    if(!in_file_pol2){
        fprintf(stderr,"Error opening Pol2 Input File\n");
        fftw_free(in);
        fftw_free(out);
        exit(1);
    }
    out_file = fopen(argv[3], "w");
    if(!out_file){
        fprintf(stderr,"Error opening output file\n");
        fftw_free(in);
        fftw_free(out);
        exit(1);
    }

    //Get number of seconds to process
    num_seconds = atoi(argv[5]);
    printf("Number of Seconds to process = %ld\n", num_seconds);

    //Get Bandwidth in MHz
    bandwidth = atoi(argv[4]);
    printf("Bandwidth = %d\n", bandwidth);

    //Calculate Beam Sampling rate
    long double sampling_rate = (long double)(1/((long double) 2*bandwidth));
    sampling_rate *= (long double) 0.000001;
    // printf("Sampling rate is = %Lf\n", (long double) sampling_rate);
    beam_sampling_rate = (long double) sampling_rate*4096;
    // printf("Beam Sampling rate is = %Lf\n", beam_sampling_rate);

    //Calculate Number of blocks from Number of Seconds
    num_blocks = (long int)(num_seconds/beam_sampling_rate);
    printf("Number of Blocks to process = %ld\n", num_blocks);

    printf("Size of output file will be %ld bytes or %ld Megabytes\n", (long int)(2*2*num_blocks*4096), (long int) ((2*2*num_blocks*4096)/(1000000)));


    //Make FFTW C2R plan
    p = fftw_plan_dft_c2r_1d(NX, in, out, FFTW_MEASURE);

    //Allocating Memory for input and output buffers
    in_stream_pol1 = (int8_t*)calloc(2*nchan, sizeof(int8_t));
    out_stream_pol1 = (int8_t*)calloc(2*2*nchan, sizeof(int8_t));
    in_stream_pol2 = (int8_t*)calloc(2*nchan, sizeof(int8_t));
    out_stream_pol2 = (int8_t*)calloc(2*2*nchan, sizeof(int8_t));

    long int lastptr, ptrnow;

    lastptr = 0;
    ptrnow = 0;
    int i = 0;
    start = time(NULL);
    while(i< num_blocks){

        //Running IFFT and filling output buffers

        //Pol1 
        read_file = fread(in_stream_pol1, sizeof(int8_t), 2*nchan, in_file_pol1);
        run_fftw(&p, in_stream_pol1, in, out, nchan);
        new_fillbuf(out, out_stream_pol1, NX);

        //Pol2
        read_file = fread(in_stream_pol2, sizeof(int8_t), 2*nchan, in_file_pol2);
        run_fftw(&p, in_stream_pol2, in, out, nchan);
        new_fillbuf(out, out_stream_pol2, NX);
    

        //Writing Out to file
        fwrite(out_stream_pol1, sizeof(int8_t), 2*2*nchan, out_file);
        fwrite(out_stream_pol2, sizeof(int8_t), 2*2*nchan, out_file);

        // progressbar(i, num_blocks);

        i++;

    }

    printf("Done!\n");

    //Freeing allocated pointers and memory
    fftw_free(in);
    fftw_free(out);
    free(in_stream_pol1);
    free(out_stream_pol1);
    free(in_stream_pol2);
    free(out_stream_pol2);

    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);

    return 0;

}