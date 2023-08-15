import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
from PyEMD import EEMD
from PyEMD import EMD

def plot_time_field(data, begin, end, sample_freq, color='r', dpi=96, figsize=(10,10), file=None):
    '''
    Plot time field image
    '''
    t = np.arange(begin, end, 1 / sample_freq)
    plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(t, data,color=color)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    if file != None:
        plt.savefig(file, bbox_inches='tight')
    #plt.show()

def plot_freq_field(data, sample_freq, dpi=96,color='r', figsize=(10,10), file=None):
    '''
    Plot frequency field image
    '''
    xf = fft.fftfreq(sample_freq, 1 / sample_freq)
    dataf = fft.fft(data, sample_freq)
    plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(xf,np.abs(dataf),color=color)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Amplitude')
    if file != None:
        plt.savefig(file, bbox_inches='tight')
    #plt.show()

def plot_emd(data, IMF, begin, end, sample_freq, figsize=(10,10), dpi=96, file=None):

    '''
    Plot EMD image
    '''
    t = np.arange(begin, end, 1 / sample_freq)
    N = IMF.shape[0]+1
    xf = fft.fftfreq(sample_freq, 1 / sample_freq) 
    dataf = fft.fft(data, sample_freq)   

    # Plot results
    plt.figure(figsize=figsize,dpi=dpi)
    plt.subplot(N, 2, 1)
    plt.plot(t, data, 'r')
    plt.title("Time Field")
    plt.ylabel("Input")
    plt.xticks([])

    plt.subplot(N, 2, 2)
    plt.plot(xf, np.abs(dataf), 'r')
    plt.title("Freqency Field")
    plt.xticks([])


    for n, imf in enumerate(IMF[0:-1]):
        plt.subplot(N, 2, (n+1)*2+1)
        plt.plot(t, imf, 'g')
        plt.ylabel("IMF "+str(n+1))
        plt.xticks([])

        plt.subplot(N, 2, (n+1)*2+2)
        imff = fft.fft(imf, sample_freq)
        plt.plot(xf, np.abs(imff), 'g')
        plt.xticks([])

    # Plot res
    plt.subplot(N, 2, 2*N-1)
    plt.plot(t, IMF[-1], 'g')
    plt.xlabel('Time [s]')
    plt.ylabel('Res')
    
    plt.subplot(N, 2, 2*N)
    resf = fft.fft(IMF[-1], sample_freq)
    plt.plot(xf, np.abs(resf), color='g')
    plt.xlabel("Frequency [Hz]")

    if file != None:
        plt.savefig(file,  bbox_inches='tight')
    #plt.show()


def eemd_denoise(data,num=[0, 1]):
    '''
    eemd denoise.
    data should be 1D squence signal.
    num is a list contain IMF need to be removed.
    '''
    IMF = EEMD().eemd(data)
    for i in num:
        IMF[i] = 0.
    
    return np.sum(IMF,axis=0)

def emd_denoise(data,num=[0, 1]):
    '''
    emd denoise.
    data should be 1D squence signal.
    num is a list contain IMF need to be removed.
    '''
    IMF = EMD().emd(data)
    for i in num:
        IMF[i] = 0.
    
    return np.sum(IMF,axis=0)

if __name__ == '__main__':
    t = np.arange(0,1,1/400)

    data = np.sin(2*np.pi*30*t)+np.cos(2*np.pi*10*t)+t*t

    plot_time_field(data, 0, 1, 400, dpi=100, file='time.jpg')
    plot_freq_field(data, 400, dpi=100, file='freq.jpg')

    emd = EMD()
    IMF = emd.emd(data,max_imf=6)
    plot_emd(data, IMF, 0, 1, 400, dpi=100, file='emd.jpg')
    data_denoise = eemd_denoise(data,num=[-1])
    plot_time_field(data_denoise, 0, 1, 400, dpi=100, file='time.jpg')
    plot_time_field(data-data_denoise, 0, 1, 400)
