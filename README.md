# gsb-ifft-pipeline
Pipeline to iFFT on GSB Voltage Beam data


# Compiling iFFT.cu

/usr/local/cuda/bin//nvcc -o ifft iFFT.cu read_file.c -L /usr/local/cuda/lib64/ -l :libcufso

# Usage

./ifft <voltage beam file> <GPU ID>
