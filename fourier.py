# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:27:13 2021

@author: luisf
"""
import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    x = np.asarray(x,dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j*np.pi*k*n/N)
    return np.dot(M,x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        #x = np.resize(a,(N + N % 2,))
        raise ValueError("must be a power of 2")
        
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])


def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")
        
    N_min = min(N, 2)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    
    while X.shape[0] < N:
            X_even = X[:, :int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2):]
            terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                            / X.shape[0])[:, None]
            X = np.vstack([X_even + terms * X_odd,
                           X_even - terms * X_odd])
    return X.ravel()

dt = 0.001
t = np.arange(0,1,dt)

f = np.sin(20*2*np.pi*t) + np.sin(90*2*np.pi*t) + np.cos(250*2*np.pi*t)
f_clean = f
f = f + 2*np.random.randn(len(t))

plt.plot(t,f,'r-',linewidth=1.5,label='noisy')
plt.plot(t,f_clean,'b-',linewidth=1.5,label='clean')
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Comparation between the noisy and clear signals')
plt.legend()
plt.show()

n = len(t)
f_fourier = np.fft.fft(f,n)
PSD = f_fourier * np.conj(f_fourier) / n
freq = (1.0 / (dt*n))*np.arange(n)
L = np.arange(1,np.floor(n/2),dtype=int)


fig, axs = plt.subplots(2,1)
fig.tight_layout()

plt.sca(axs[0])
plt.plot(t,f_clean,'b-',linewidth=1.5,label='clean')
plt.plot(t,f,'r-',linewidth=1.5,label='noisy')
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Comparation between the noisy and clear signals')
plt.ylim(-6,6)
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L],np.abs(PSD[L]),'g--',linewidth=1.5,label='PSD')
plt.grid()
plt.xlabel('f')
plt.ylabel('PSD')
plt.title('Power Spectrum Density along the frequency axis')
plt.ylim(0,350)
plt.legend()
plt.show()


indices = PSD > 100
PSD_clean = PSD * indices
f_fourier = f_fourier*indices
f_inv = np.fft.ifft(f_fourier)

fig, axs = plt.subplots(2,1)
fig.tight_layout()

plt.sca(axs[0])
plt.plot(t,f_clean,'b-',linewidth=1.5,label='original')
plt.plot(t,f_inv,'r-',linewidth=1.5,label='restored')
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Comparation between the original and restored signals')
plt.ylim(-2.5,2.5)
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L],np.abs(PSD_clean[L]),'g-',linewidth=1.5,label='PSD')
plt.grid()
plt.xlabel('f')
plt.ylabel('PSD')
plt.title('Clean Power Spectrum Density along the frequency axis')
plt.ylim(0,350)
plt.legend()
plt.show()