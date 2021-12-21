from os import NGROUPS_MAX
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import _join_dispatcher 
import time
import functools
import cupy as cp
from threading import Thread
from multiprocessing import Process
import concurrent.futures
from numba import njit, prange

NX = 8
# a = array("h")

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

def create_array(data):
        num_array = [np.int8(numeric_string) for numeric_string in data]
        print(num_array[0])
        #print(num_array[:5])
        P = [num_array[2*i] + num_array[2*i+1]*1j for i in range(NX)]
        P = cp.array(P, dtype=complex) 
        # P = np.array(P, dtype=complex)
        print(P.device)
        return P

def run_cuFFT(f,I,N):
        print("In CUFFT")
        print(I,N)
        cp.cuda.Device(1).use()
        # f = open(fileName,'rb')
        # h = open('thread1.txt','w')
        # iffts = []
        # cp.fft.config.use_multi_gpus = True
        # cp.fft.config.set_cufft_gpus([2,3]) #use GPUs 0 & 1
        # with open(fileName,'rb') as f:
                
        for i in range(I,I+N):
                f.seek(i*NX)
                DATA = f.read(2*NX)
                print("cuFFT Data Block "+ str(i)+ "\n")
                print(I,N)
                        # print(DATA[:5])
                print("\n")
                P = create_array(DATA)
                # ifft = iFFT(P)
                # ifft = np.array(ifft)
                ifft = cp.fft.ifft(P)
                # ifft1 = cp.fft.ifft(P[:NX/4])
                # ifft2 = cp.fft.ifft(P[NX/4:NX/2])
                # ifft3 = cp.fft.ifft(P[NX/2:3*NX/4])
                # ifft4 = cp.fft.ifft(P[3*NX/4:]) 
                # ifft = cp.concatenate([ifft1,ifft2,ifft3,ifft4])
                        # ifft = cp.asnumpy(ifft)
                        # iffts.append(ifft)
                        # h.write(str(ifft))
                        # ifft = iFFT(P)
                        # ifft = np.array(ifft)/NX
                print(ifft[:5])
                        # print(np.shape(ifft)[0])
                print('\n')
        # h.close()
        # iffts = np.array(iffts)
        # iffts = np.ravel(iffts)
        # plt.plot(iffts)
        # plt.show()
        # plt.savefig("Sample_Timeseries.png")


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
        n = 1
        n_processes = 1
        fileName = 'B0740-28_B4_25MHz_P1_small.rawvlt'
        f = open(fileName,'rb')

        t = [0]*n_processes
        for i in range(n_processes):
                t[i] = Process(target = run_cuFFT, args = (f,i*n,n))

        for i in range(n_processes):
                t[i].start()

        for i in range(n_processes):
                t[i].join()
                print("T"+str(i+1)+" is done")
        
        print("Done!")
        


        



if __name__ == "__main__":
        main()
