# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 12:34:49 2021

@author: luisf
"""

import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
import sounddevice as sd

def filter(x,a,b):
    output = np.zeros(len(x))
    for n in range(4,len(x),1):
        output[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + b[3]*x[n-3] + b[4]*x[n-4] - a[1]*output[n-1] - a[2]*output[n-2] - a[3]*output[n-3] - a[4]*output[n-4]
        output[n] = output[n] / a[0]
    return output

#design IIR Butterworth filter
Fs = 1000#Hz
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


f1 = 90#Hz
f2 = 10#Hz
f3 = 180#Hz
dt = 1 /Fs
t = np.arange(0,1,dt)
signal = np.sin(2*np.pi*t*f1) + np.sin(2*np.pi*t*f2) + np.cos(2*np.pi*t*f3)
sd.play(4*signal,Fs)

plt.plot(t,signal,'r-',label='Original Signal')


filteredSignal = filter(signal,a,b)
plt.plot(t,filteredSignal,'b-',label='Filtered Signal')
plt.xlabel('time')
plt.ylabel('signal')
plt.title('Original Signal')
plt.grid()
plt.legend()
plt.show()
# sd.play(4*filteredSignal,Fs)