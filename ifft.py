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
queue = Queue()
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

def save_file(file, return_dict):
        ifft_dict = {}

        for i in sorted(return_dict):
                ifft_dict[i] = return_dict[i]

        print("iift_dict keys")
        print(ifft_dict.keys())
        # file = open(filename, 'wb')
        for key in ifft_dict.keys():
                print("Writing for Block ", key)
                ifft_blocks = ifft_dict[key]
                for i in range(len(ifft_blocks)//2048):
                        ifft_block = ifft_blocks[i*NX: i*NX + NX]
                        # ifft_r = ifft_block[::2]
                        # print(ifft_r[:5])
                        # ifft_i = ifft_block[1::2]
                        ifft_r = []
                        ifft_i = []
                        for i in range(np.shape(ifft_block)[0]):
                                ifft_r.append(ifft_block[i].real)
                                ifft_i.append(ifft_block[i].imag)

                        ifft_r = np.clip(ifft_r,-128,127)
                        ifft_i = np.clip(ifft_i,-128,127)
                        final_ifft_r = []
                        final_ifft_i = []
                        for i in range(np.shape(ifft_r)[0]):
                                final_ifft_r.append(int(ifft_r[i]))
                                final_ifft_i.append(int(ifft_i[i]))
                        for i in range(len(final_ifft_r)):
                                file.write(final_ifft_r[i].to_bytes(1,byteorder='big',signed=True))
                                file.write(final_ifft_i[i].to_bytes(1,byteorder='big',signed=True))
        file.close()

        

def run_cuFFT(f,I,N,return_dict):
        print("In CUFFT")
        print(I,N)
        cp.cuda.Device(1).use()
        # f = open(fileName,'rb')
        # h = open('thread1.txt','w')
        iffts = []
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
                ifft = cp.asnumpy(ifft)
                iffts.append(ifft)
                        # h.write(str(ifft))
                        # ifft = iFFT(P)
                        # ifft = np.array(ifft)/NX
                print(ifft[:5])
                        # print(np.shape(ifft)[0])
                print('\n')
        # h.close()
        iffts = np.array(iffts)
        iffts = np.ravel(iffts)
        # plt.plot(iffts)
        # plt.show()
        # plt.savefig("Sample_Timeseries.png")
        # queue.put({I:iffts})
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
        n = 3051
        n_processes = 40
        # fileName = 'B0740-28_B4_25MHz_P1_small.rawvlt'
        fileName = 'pulsar_2s.vlt'
        f = open(fileName,'rb')
        # queue = Queue()
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
        print("Keys")
        print(return_dict.keys())
        print("Values")
        print(np.shape(return_dict.values()[0]))

        print("Writing to File")
        
        h = open("pulsar_2s_timeseries.raw",'wb')
        save_file(h,return_dict)
        h.close()
        f.close()
        


        



if __name__ == "__main__":
        main()
