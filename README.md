# gsb-ifft-pipeline
Pipeline to iFFT on GSB Voltage Beam data


# Compiling iFFT.cu on BLPC1

/usr/local/cuda/bin//nvcc -o ifft iFFT.cu read_file.c -L /usr/local/cuda/lib64/ -l :libcufso

nvcc -o ifft_stream iFFT_stream.cu read_file.c -L /usr/local/cuda/lib64/ -l :libcufft.so -std=c++11

# Usage

## Cuda 
./ifft voltage-beam-file-name GPU ID

./ifft_stream voltage-beam-file-name output-file-name GPU ID

## Python
### Requirements
Python 3.8+
Cupy
Numpy

# Simulated Pulsar Timeseries

A one second grab of simulated pulsar timeseries with a sampling rate of 20 nanoseconds

![Simulated Timeseries](https://github.com/RaghavGirgaonkar/gsb-ifft-pipeline/blob/main/images/simulated_timeseries.png?raw=true)

# Regenerated Pulsar Timeseries

A one second grab of regenarated pulsar timeseries using ifft.py

![Regenerated Timeseries](https://github.com/RaghavGirgaonkar/gsb-ifft-pipeline/blob/main/images/regenerated_timeseries.png?raw=true)
