# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:24:39 2021

@author: luisf
"""
import numpy as np 
import sounddevice as sd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=[12,8]
plt.rcParams.update({'font.size':18})

dt = 0.001
t  = np.arange(0,2,dt)
f0 = 50
f1 = 250
t1 = 2

x = np.cos(2*np.pi*t*(f0 + (f1-f0)*np.power(t,2)/(3*t1**2)))
print(x.shape)
fs = 1 / dt
sd.play(2*x,fs)
spec,frequencies,t2,im = plt.specgram(x,NFFT=128,Fs=fs,noverlap=120,cmap='jet_r')
print(spec.shape,frequencies.shape,t2.shape)
plt.title('Spectogram')
plt.colorbar()
plt.show()


fig, axs = plt.subplots(2,1)
fig.tight_layout()

plt.sca(axs[0])
plt.plot(t,x)
plt.xlabel('time')
plt.ylabel('f(t)')
plt.title('Function in time domain')
plt.grid()

n = int(len(t))
x_fft = np.fft.fft(x,n)
PSD   = x_fft * np.conj(x_fft) / n
freq  = (1.0 / (dt*n))*np.arange(n)
L = np.arange(1,np.floor(n/2),dtype=int)

plt.sca(axs[1])
plt.plot(freq[L],np.abs(PSD[L]),'r-',linewidth=1.5,label='PSD')
plt.xlabel('freq')
plt.ylabel('PSD')
plt.title('Function in frequency domain')
plt.grid()
plt.show()

def spectogram(x,NFFT,overlap):
    starts = np.arange(0,len(x),NFFT-overlap,dtype='int')
    starts = starts[starts+NFFT < len(x)]
    xInput = []
    for start in starts:
        window = np.fft.fft(x[start:start+NFFT],NFFT)
        xInput.append(window)
    
    spec = np.array(xInput).T
    spec = 10*np.log10(spec)
    #assert(spec.shape[0] == len(starts))
    return (starts,spec)

# a,b = spectogram(x,128,120)
# plt.imshow(np.abs(b),origin='lower')