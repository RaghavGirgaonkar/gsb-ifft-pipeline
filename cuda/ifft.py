from os import NGROUPS_MAX
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import _join_dispatcher 
import time
import functools
import cupy as cp
from threading import Thread
from multiprocessing import Process, Queue, Manager
import concurrent.futures
from numba import njit, prange

NX = 2048

def timefunc(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure

def complete_spectrum(ifft_block):
        """
        Using the 2048 complex points made, the spectrum is completed by flipping the 
        positive frequency coefficients and taking their complex conjugate

        """
        ifft_conjugate = ifft_block[:0:-1].conjugate()
        ifft_array = np.concatenate((ifft_conjugate,ifft_block))
        return ifft_array


def create_array(data):
        """
        Creates a complex array of 2048 length after reading 4096 points from the 
        raw voltage beam file

        """
        num_array = [np.int8(numeric_string) for numeric_string in data]
        P = [num_array[2*i] + num_array[2*i+1]*1j for i in range(NX)]
        P = cp.array(P, dtype=complex) 
        print(P.device)
        return P

def save_file(file, ifft_dict):
        """
        Function to save iFFT data to a binary file: Clip ifft array between -128, 127 --> convert to int --> 
        convert to bytes --> store in binary file

        """
        for key in ifft_dict.keys():
                print("Writing for Block ", key)
                ifft_blocks = ifft_dict[key]
                for i in range(len(ifft_blocks)//NX):
                        ifft_block = ifft_blocks[i*NX: i*NX + NX]
                        ifft_r = []
                        ifft_i = []
                        for i in range(np.shape(ifft_block)[0]):
                                ifft_r.append(ifft_block[i].real)
                                ifft_i.append(ifft_block[i].imag)

                        ifft_r = np.clip(ifft_r,-128,127)
                        ifft_i = np.clip(ifft_i,-128,127)
                        final_fft = []
                        for i in range(np.shape(ifft_r)[0]):
                                final_fft.append(ifft_r[i])
                                final_fft.append(ifft_i[i])
                        final_fft = np.array(final_fft,dtype=int)
                        binary_ifft_array = np.array(final_fft, dtype = np.int8).tobytes()
                        file.write(binary_ifft_array)
        file.close()

        

def run_cuFFT(f,I,N,return_dict):
        """
        Runs iFFT on multiple FFT blocks of raw data using cuFFT
        Stores output in a shared dictionary between multiple processes

        """
        print("In CUFFT")
        print(I,N)
        cp.cuda.Device(1).use()

        iffts = []
                
        for i in range(I,I+N):
                f.seek(i*2*NX)
                DATA = f.read(2*NX)
                print("cuFFT Data Block "+ str(i)+ "\n")
                print(I,N)
                print("\n")
                P = create_array(DATA)
                ifft_array = complete_spectrum(P)
                ifft = cp.fft.ifft(ifft_array)
                ifft = cp.asnumpy(ifft)
                iffts.append(ifft)
                print(ifft[:5])
                print('\n')

        iffts = np.array(iffts)
        iffts = np.ravel(iffts)
        return_dict[I] = iffts



#Custom iFFT routine
def iFFT(P):
        n = np.shape(P)[0]
        if n == 1:
                return P
        w = (1/n)*(np.cos(2*np.pi/n) - 1j*np.sin(2*np.pi/n))
        P_e, P_o = P[::2], P[1::2]
        y_e, y_o = iFFT(P_e), iFFT(P_o)
        y = [0]*n
        for k in range(int(n/2)):
                y[k] = y_e[k] + w**k*y_o[k]
                y[k + int(n/2)] = y_e[k] - w**k*y_o[k]
        return y



@timefunc
def main():
        n = 1252
        n_processes = 39
        IN_FILE = "pulsar_sample.vlt"
        OUT_FILE = "pulsar_sample.raw"
        f = open(IN_FILE,'rb')
        manager = Manager()
        return_dict = manager.dict()
        t = [0]*n_processes
        for i in range(n_processes):
                t[i] = Process(target = run_cuFFT, args = (f,i*n,n, return_dict))

        for i in range(n_processes):
                t[i].start()

        for i in range(n_processes):
                t[i].join()
                print("T"+str(i+1)+" is done")
        
        print("Done!")

        print("Shape of Values")
        print(np.shape(return_dict.values()[0])) #Should be n*shape_of_one_ifft_block (in this case: 4095)
        ifft_dict = {}

        for i in sorted(return_dict):
                ifft_dict[i] = return_dict[i]

        print("iift_dict keys")
        print(ifft_dict.keys())

        print("Writing to File")
        
        h = open(OUT_FILE,'wb')

        save_file(h,ifft_dict)
        f.close()
        


        



if __name__ == "__main__":
        main()
