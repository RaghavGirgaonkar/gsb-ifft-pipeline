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
#include "guppi_header.h"
#include <omp.h>


#define LINUX
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define NCHAN 2048

void get_spectrum(int8_t *pol1_data, int8_t *spectrum){

    // omp_set_num_threads(8);

    int8_t Nby2 = pol1_data[1];
    int8_t Nby2_imag = 0;

    // #pragma omp parallel
    for(int i = 0; i < NCHAN*2 - 2; i++){
        spectrum[i] = pol1_data[i+2];
    }


    spectrum[NCHAN*2 - 2] = Nby2;
    spectrum[NCHAN*2 - 1] = Nby2_imag;

}

void file_extension(int j, char* root){
    char extension[128];

    if(j<10){
        snprintf(extension, sizeof(extension), ".000%d.raw", j);
    }
    else if(j >= 10 && j <100){
        snprintf(extension, sizeof(extension), ".00%d.raw", j);
    }
    else if(j >= 100 && j <1000){
        snprintf(extension, sizeof(extension), ".0%d.raw", j);
    }
    else{
        snprintf(extension, sizeof(extension), ".%d.raw", j);
    }

    strcpy(root,extension);

}

int main(int argc, char* argv[]){

    time_t start, stop;

    // omp_set_num_threads(8);

    if(argc < 7)
    { fprintf(stderr, "USAGE: %s <Input File Pol1> <Input File Pol2> <Output File Stem> <Bandwidth of Observation in MHz> <Start Frequency in MHz> <Num Seconds>\n", argv[0]);
      return(-1);
    }

    FILE *pol1_file, *pol2_file, *guppi_header_file, *out_file;
    int samples_per_frame = 32768;
    int max_BLOCS_per_file = 3;
    long int BLOCSIZE = NCHAN*2*samples_per_frame; // About 134 MiB per BLOCK
    char* guppi_header_data;
    int bandwidth;
    int num_seconds;
    int read_file;
    char output_file[100];
    

    int8_t *pol1_data, *pol2_data, *pol1_spectrum, *pol2_spectrum;

    //Open files
    pol1_file = fopen(argv[1], "r");
    if(!pol1_file){
        fprintf(stderr,"Error opening Pol1 Input File\n");
        exit(1);
    }

    pol2_file = fopen(argv[2], "r");
    if(!pol2_file){
        fprintf(stderr,"Error opening Pol1 Input File\n");
        exit(1);
    }

    //Output files
    
    // strcat(output_file, ".0000.raw");
    
    // out_file = fopen(output_file, "w");
    // if(!out_file){
    //     fprintf(stderr,"Error opening output file\n");
    //     exit(1);
    // }
    // else{
    //     printf("Created Output File: %s\n", output_file);
    // }

    //Get Bandwidth in MHz
    bandwidth = atoi(argv[4]);
    printf("Bandwidth = %d\n", bandwidth);

    int start_freq  = atoi(argv[5]);
    printf("Starting Frequency is = %d\n", start_freq);

    //Setting OBS_BW, OBSFREQ and CHAN_BW in header_template file

    //OBS_BW
    update_header_param("OBSBW", "f", (long double) bandwidth);

    //CHAN_BW
    update_header_param("CHAN_BW", "Lf", ((long double)bandwidth/(long double)NCHAN));

    //OBSFREQ
    update_header_param("OBSFREQ", "f", (long double) ((long double)start_freq + (long double)((long double)bandwidth/((long double)2))));



    //Setting DIRECTIO to 1
    update_header_param("DIRECTIO","i",(long double) 1);

    //(TEST) Update Observer name
    update_header_param("OBSERVER", "Arun M", 0);

    //(TEST) Update BANKNAM name
    update_header_param("BANKNAM", "GWBH8", 0);

    //Updating NPOL in Header File
    update_header_param("NPOL", "i", (long double)4);

    //Get num seconds
    num_seconds = atoi(argv[6]);
    printf("Number of seconds to process = %d\n", num_seconds);


    //Get total number of blocks
    long int num_blocks;
    long double sampling_rate = (long double)(1/((long double) 2*bandwidth));
    sampling_rate *= (long double) 0.000001;
    long double beam_sampling_rate = (long double) sampling_rate*4096;
    num_blocks = (long int)(num_seconds/beam_sampling_rate);

    printf("The beam sampling rate is %.10Lf\n", beam_sampling_rate);
  
    update_header_param("TBIN", "Lf", beam_sampling_rate);

    printf("Total number of blocks to process per polarisation file = %ld\n", num_blocks);

    int num_BLOCS = (int) 2*(num_blocks/samples_per_frame);

    //Calculate number of files that will be made
    int num_files;

    if(num_BLOCS%max_BLOCS_per_file == 0) num_files = (int) ((num_BLOCS/max_BLOCS_per_file));
    else num_files = (int) ((num_BLOCS/max_BLOCS_per_file) + 1);
    printf("Number of files that will be created = %d\n",num_files);

    printf("TotaL Number of Headers that will be written is = %d and the total time length of all files together will be = %Lf seconds\n", num_BLOCS, (long double)(num_BLOCS*(samples_per_frame/2)*beam_sampling_rate));


    
    // printf("Making BLOCK...\n");
    int8_t **BLOCK = (int8_t**)malloc(NCHAN*sizeof(int8_t*));
    for(int i = 0; i < NCHAN; i++){
        BLOCK[i] = (int8_t*)malloc(2*samples_per_frame*sizeof(int8_t));
    }
    // printf("Done\n");
    pol1_data = (int8_t*)calloc(2*NCHAN, sizeof(int8_t));
    pol2_data = (int8_t*)calloc(2*NCHAN, sizeof(int8_t));
    pol1_spectrum = (int8_t*)calloc(2*NCHAN, sizeof(int8_t));
    pol2_spectrum = (int8_t*)calloc(2*NCHAN, sizeof(int8_t));
    start = time(NULL);

    
    int f = 0;
    int BLOCS;

    while(f < num_files){

        if(num_BLOCS < max_BLOCS_per_file) BLOCS = num_BLOCS;
        else BLOCS = max_BLOCS_per_file;

        printf("BLOCS = %d\n", BLOCS);

        //Create File extension
        strcpy(output_file, argv[3]);
        char *root = (char *)malloc(50);
        file_extension(f,root);
        strcat(output_file, root);
        printf("Creating File %s\n",output_file);

        //Create Output File
        out_file = fopen(output_file, "w");
        if(!out_file){
            fprintf(stderr,"Error opening output file\n");
            exit(1);
        }
        else{
            printf("Created Output File: %s\n", output_file);
        }

        //Update SCANLEN
        update_header_param("SCANLEN", "Lf", (long double)(BLOCS*(samples_per_frame/2)*beam_sampling_rate));

        //Make GUPPI Header

        //Make Header File
        int gh = make_guppi_header();
        if(!gh){
            printf("Making Guppi Header File failed\n");
            exit(1);
        }

        //Get GUPPI RAW Header file
        const char * guppi_header_path = "guppi_header.txt";
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

        //Size of File 
        printf("Size of output file will be %ld bytes or %ld Megabytes\n", (long int) (BLOCS)*(BLOCSIZE + (header_size)),(long int) (((BLOCS)*(BLOCSIZE + (header_size)))/(1000000)));


        //Write GUPPI RAW File
        int i = 0;
        while(i < BLOCS){

            for(int j = 0; j < (int) samples_per_frame/2; j++){

                read_file = fread(pol1_data, sizeof(int8_t), 2*NCHAN, pol1_file);
                read_file = fread(pol2_data, sizeof(int8_t), 2*NCHAN, pol2_file);
                get_spectrum(pol1_data, pol1_spectrum);
                get_spectrum(pol2_data, pol2_spectrum);

                //Populate BLOCK
                for(int k = 0; k < NCHAN; k++){

                    BLOCK[k][4*j] = pol1_spectrum[2*k];
                    BLOCK[k][4*j+1] = pol1_spectrum[2*k+1];
                    BLOCK[k][4*j+2] = pol2_spectrum[2*k];
                    BLOCK[k][4*j+3] = pol2_spectrum[2*k+1];
                    
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

        f += 1;
        num_BLOCS -= max_BLOCS_per_file;
        free(guppi_header_data);
        free(root);
        fclose(out_file);
        fclose(guppi_header_file);
    }
    
    //Free Memory and Close Files
    for(int i = 0; i < NCHAN; i++){
        free(BLOCK[i]);
    }
    free(BLOCK);
    free(pol1_data);
    free(pol2_data);
    free(pol1_spectrum);
    free(pol2_spectrum);
    fclose(pol1_file);
    fclose(pol2_file);
    stop = time(NULL);
    printf("The number of seconds for to run was %ld\n", stop - start);


    return 0;
}
