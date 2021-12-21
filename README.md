# gsb-ifft-pipeline
Pipeline to iFFT on GSB Voltage Beam data


#Compiling iFFT.cu

nvcc -o ifft iFFT.cu read_file.c -L /usr/local/cuda/lib64/ -l :libcufso
