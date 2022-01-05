import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="Generate High SNR Pulsar Timeseries and store data in Spectral blocks")
parser.add_argument('-t','--time', type=int, help="Time in seconds of total timeseries")
parser.add_argument('-nc','--nchan', type=int, help="Number of frequency channels for spectral block")
parser.add_argument('-a','--amplitude', type=int, help="Constant to multiply Gaussian of pulse (to increase/ decrease peak value)")
parser.add_argument('-pp','--pulsarperiod', type=int, help="Pulsar Period in seconds (Cannot be > Total time)")
parser.add_argument('-s','--beam_sampling_rate', type=float, help="Beam Sampling rate in us")
parser.add_argument('-o','--output', type=str, help="Output file name")
args = parser.parse_args()

def gaussian(x,mu,C):
    numerator = np.exp((-1*(x - mu)**2)/2)
    denominator = np.sqrt(2*np.pi)
    gaussian = C*(numerator/denominator)
    return gaussian

def whitenoise(N_samples, mu = 0, stdv = 1):
    white_noise = np.random.normal(mu,stdv, N_samples)
    return white_noise

def create_gaussian_series(psr_period, sampling_rate, N_samples, C):
    #Position of first pulse
    p_pulse = int(psr_period/sampling_rate)
    print("Position of first pulse ", p_pulse)
    #Number of pulses
    num_pulses = int(N_samples*sampling_rate)//psr_period
    print("Number of pulses ", num_pulses)
    #Generate first pulse
    N = int(2*p_pulse)
    x = np.linspace(0,N,N)
    mu = N//2
    p1 = gaussian(x,mu, C)
    print("Generated one pulse")
    #Generate rest of the pulses
    pulse_train = p1
    for i in range(1,num_pulses+1):
        # d = np.linspace(i*N, (i+1)*N, N)
        # mu = N//2 + i*N
        # p = gaussian(d,mu, C)
        print("Generating pulse number ", i+1)
        pulse_train = np.concatenate((pulse_train,p1))

    return pulse_train[:N_samples]

def get_fft(timeseries, nchan, N_samples):
    Nx = 2*nchan
    fft_r = []
    fft_i = []
    print("Finding FFT ")
    for i in range(N_samples//Nx):
        temp_timeseries = timeseries[i*Nx:i*Nx + Nx]
        fft_p = np.fft.fft(temp_timeseries)
        #Combining the DC and N/2 term into a single complex number which goes at the start of the FFT block
        DC_term = fft_p[0].real
        Nby2_term = fft_p[int(Nx/2)].real
        first_fft_term = np.array([DC_term + Nby2_term*1j])
        fft_pulsar = np.concatenate((first_fft_term, fft_p[1:int(Nx/2)])) #Take positive freqs and the modifed first freq term
        #Separate Real and Imaginary Values
        for ff in fft_pulsar:
            fft_r.append(ff.real)
            fft_i.append(ff.imag)

    #Clip fft values
    print("Clipping...")
    fft_r = np.clip(fft_r,-128,127)
    fft_i = np.clip(fft_i,-128,127)

    #Create final FFT array
    print("Creating Final FFT Array...")
    final_fft = []
    for i in range(np.shape(fft_r)[0]):
        final_fft.append(fft_r[i])
        final_fft.append(fft_i[i])
    final_fft = np.array(final_fft,dtype=int)

    return final_fft

def save_file(filename, final_fft):
    file = open(filename,'wb')
    binary_fft_array = np.array(final_fft, dtype = np.int8).tobytes()
    file.write(binary_fft_array)
    file.close()




def main():

    #Get arguements
    total_time = args.time
    nchan = args.nchan
    C = args.amplitude
    pulsar_period = args.pulsarperiod
    beam_sampling_rate = args.beam_sampling_rate
    filename = args.output

    #Check 
    if pulsar_period >= total_time:
        print("TOTAL TIME ", total_time)
        print("PSR PERIOD ", pulsar_period)
        sys.exit("PSR Period cannot be greater than Total Time")

    #Display Arguments
    print("TOTAL TIME(s) ", total_time)
    print("NCHAN ", nchan)
    print("AMPLITUDE ", C)
    print("PSR PERIOD(s) ", pulsar_period)
    print("BEAM SAMPLING RATE(us) ", beam_sampling_rate)
    print("OUTPUT FILE ", filename)

    #Calculate Nyquist Sampling rate
    sampling_rate = beam_sampling_rate/(2*nchan*1000000)
    print("Sampling rate is = ", sampling_rate)

    #Calculate number of samples
    N_samples = int(total_time/sampling_rate)
    print("Total number of samples = ", N_samples)

    #Create White Noise
    print("Creating White Noise")
    white_noise = whitenoise(N_samples, 0, 1)

    #Create Pulsar Train
    print("Creating Pulse Train")
    pulse_train = create_gaussian_series(pulsar_period, sampling_rate, N_samples, C)

    #Create total Timeseries
    print("Creating Total Timeseries")
    timeseries = white_noise + pulse_train

    #Calculate FFT
    print("Running FFT")
    final_fft = get_fft(timeseries, nchan, N_samples)

    #Write to file
    print("Writing to file " + filename)
    save_file(filename, final_fft)





if __name__ == "__main__":
    main()