# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:46:36 2021

@author: luisf
"""
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

def filter(x,a,b):
    output = np.zeros(len(x))
    for i in range(3,len(x),1):
        output[i] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + b[3]*x[n-3] -a[0] - a[1]*output[i-1] - a[2]*output[i-2]- a[3]*output[i-3]
    return output

#design IIR Butterworth filter
Fs = 360#Hz
n  = 4 
fc = 45#Hz
w_c = 2*fc/Fs#normalized Frequencies

b,a = sig.butter(n,w_c)

#Frequency response
w,h = sig.freqz(b,a,worN=2000)
w = Fs*w/(2*np.pi)
h_db = 20*np.log10(abs(h))
h_phase = np.angle(h)

fig, axs = plt.subplots(2)
fig.tight_layout()


plt.sca(axs[0])
plt.title('Bode Plot IIR - Low Pass Filter Butterworth')
plt.plot(w,h_db)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude(dB)')
plt.xlim(0, Fs / 2)
plt.ylim(-80, 10)
plt.axvline(fc, color='red')
plt.axhline(-3, linewidth=0.8, color='black', linestyle=':')
plt.grid()


h_deg = np.angle(h)
h_deg = np.unwrap(h_deg)
h_deg = np.rad2deg(h_deg)

plt.sca(axs[1])
plt.plot(w,h_deg)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (º)')
plt.grid()
plt.xlim(0, Fs / 2)
plt.ylim(-360, 0)
plt.axvline(fc, color='red')
plt.show()

#design FIR filter
N   = 12
t   = sig.firwin(N,w_c)
w,h = sig.freqz(t,worN=2000)

w     = Fs*w/(2*np.pi)
h_db  = 20*np.log10(abs(h))
h_deg = np.angle(h)
h_deg = np.unwrap(h_deg)
h_deg = np.rad2deg(h_deg)

fig, axs = plt.subplots(2)
fig.tight_layout()
plt.sca(axs[0])
plt.title('Bode Plot FIR - Low Pass Filter')
plt.plot(w,h_db)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude(dB)')
plt.xlim(0, Fs / 2)
plt.ylim(-80, 10)
plt.axvline(fc, color='red')
plt.axhline(-3, linewidth=0.8, color='black', linestyle=':')
plt.grid()


plt.sca(axs[1])
plt.plot(w,h_deg)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (º)')
plt.grid()
plt.xlim(0, Fs / 2)
plt.ylim(-360, 0)
plt.axvline(fc, color='red')
plt.show()

plt.show()
