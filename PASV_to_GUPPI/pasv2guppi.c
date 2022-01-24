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
#include <sys/stat.h>


#define LINUX
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define NCHAN 2048

void get_spectrum(int8_t *pol1_data, int8_t *spectrum){

    int8_t Nby2 = pol1_data[1];
    int8_t Nby2_imag = 0;

    for(int i = 0; i < NCHAN*2 - 2; i++){
        spectrum[i] = pol1_data[i+2];
    }
    spectrum[NCHAN*2 - 2] = Nby2;
    spectrum[NCHAN*2 - 1] = Nby2_imag;

}

int main(int argc, char* argv[]){

    time_t start, stop;

    if(argc < 4)
    { fprintf(stderr, "USAGE: %s <Input File Pol1> <Output File> <Bandwidth of Observation in MHz>\n", argv[0]);
      return(-1);
    }

    FILE *pol1_file, *guppi_header_file, *out_file;
    int samples_per_frame = 4096;
    long int BLOCSIZE = NCHAN*2*samples_per_frame; // About 50 MiB per BLOCK
    char* guppi_header_data;
    int bandwidth;
    int read_file;

    int8_t *pol1_data, *spectrum;

    //Open files
    pol1_file = fopen(argv[1], "r");
    if(!pol1_file){
        fprintf(stderr,"Error opening Pol1 Input File\n");
        exit(1);
    }
    out_file = fopen(argv[2], "w");
    if(!out_file){
        fprintf(stderr,"Error opening output file\n");
        exit(1);
    }

    //Get GUPPI RAW Header file
    const char * guppi_header_path = "guppi_header_template_B0740.txt";
    guppi_header_file = fopen(guppi_header_path, "r");
    if(!guppi_header_file){
          fprintf(stderr,"Error opening GUPPI HEADER File\n");
          exit(1);
      }

    //Get size of GUPPI Header
    struct stat st;
    stat(guppi_header_path, &st);
    size_t header_size = st.st_size;

    printf("Size of GUPPI Header is %zu bytes\n", header_size);

    //Store GUPPI Header in a buffer
    guppi_header_data = (char*)malloc(header_size);
    int g = fread(guppi_header_data, sizeof(char), header_size, guppi_header_file);

    //Get Bandwidth in MHz
    bandwidth = atoi(argv[3]);
    printf("Bandwidth = %d\n", bandwidth);

    //Get size of Input File
    stat(argv[1], &st);
    size_t file_size = st.st_size;

    long int num_blocks = 65536;

    printf("Size of file is %zu bytes...number of FFT Blocks (total number of samples) is %ld\n", num_blocks*2*NCHAN, num_blocks);

    int i = 0;
    printf("Making BLOCK...\n");
    int8_t **BLOCK = (int8_t**)malloc(NCHAN*sizeof(int8_t*));
    for(int i = 0; i < NCHAN; i++){
        BLOCK[i] = (int8_t*)malloc(2*samples_per_frame*sizeof(int8_t));
    }
    printf("Done\n");
    pol1_data = (int8_t*)calloc(2*NCHAN, sizeof(int8_t));
    spectrum = (int8_t*)calloc(2*NCHAN, sizeof(int8_t));
    printf("Size of output file will be %ld bytes or %ld Megabytes\n", (long int) (num_blocks/samples_per_frame)*(BLOCSIZE + header_size),(long int) (((num_blocks/samples_per_frame)*(BLOCSIZE + header_size))/(1000000)));
    start = time(NULL);
    //Write GUPPI File
    while(i < (int) (num_blocks/samples_per_frame)){

        printf("I = %d\n", i);

        for(int j = 0; j < samples_per_frame; j++){

            read_file = fread(pol1_data, sizeof(int8_t), 2*NCHAN, pol1_file);
            get_spectrum(pol1_data, spectrum);

            //Populate BLOCK
            for(int k = 0; k < NCHAN; k++){

                BLOCK[k][2*j] = spectrum[2*k];
                BLOCK[k][2*j+1] = spectrum[2*k+1];
                // printf("Populating BLOCK[%d][%d] and BLOCK[%d][%d]\n", k, 2*j, k, 2*j+ 1);
            }

        }

        //Write GUPPI Header
        fwrite(guppi_header_data, sizeof(char), header_size, out_file);
        //Write BLOCK
        // fwrite(BLOCK, 2*samples_per_frame*sizeof(int8_t), NCHAN, out_file);
        for(int l = 0; l < NCHAN; l++){
            fwrite(BLOCK[l], sizeof(int8_t), 2*samples_per_frame, out_file);
        }

        i++;



    }
    
    //Free Memory and Close Files
    for(int i = 0; i < NCHAN; i++){
        free(BLOCK[i]);
    }
    free(BLOCK);
    free(guppi_header_data);
    free(pol1_data);
    free(spectrum);
    fclose(pol1_file);
    fclose(out_file);
    fclose(guppi_header_file);
    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);


    return 0;
}