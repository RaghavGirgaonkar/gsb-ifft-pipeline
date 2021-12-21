#include "read_file.h"
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <complex.h>

#define NX 2048

char *read_file(const char *filename){
    FILE* buffer = fopen(filename,"rb");
    struct stat sb;
    if (stat(filename, &sb) == -1) {
        perror("stat");
        exit(EXIT_FAILURE);
    }
    printf("Size of file is %zu\n", sb.st_size);

    //Reading file into buffer
    size_t file_size = sb.st_size;
    size_t n = 1;
    char* file_contents = malloc(file_size);

    fread(file_contents, sb.st_size, n, buffer);
    // for(int i = 4096; i< 4096+4096; i++){
    //     printf("%d", file_contents[i]);
    // }
    return file_contents;
    
    
    // int op_file;
    // char *P;
    // // int arr[2*NX];
    // if((P = malloc(data_size)) == NULL) {perror("Memory allocation error");exit(0);}
    // op_file = open (c, O_RDONLY);
    // if (op_file == -1) { fprintf (stderr, "writing record file %s \n", c); exit (EXIT_FAILURE); }
    // if(read(op_file, P, data_size) != data_size ) {perror ("Read Error op_file0:"); exit(0);}
    // // printf("Reading 100 enties of data file:\n");
    // // for(int i = index; i< index + 2*NX;i++){
    // //     arr[i - index] = P[i];
    // //     printf("%d\n",P[i]);
    // // }
    // // int len = *(&P + 1) - P;
    // // printf("Length of data is %d\n",len);
    // return P;
}
