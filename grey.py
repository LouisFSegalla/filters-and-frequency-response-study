# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 23:25:19 2021

@author: luisf
"""
import numpy as np
k=2
kern=np.ones(2*k+1)/(2*k+1)
arr=np.random.random((5))
kern_fft = np.fft.fft(kern)
arr_fft = np.fft.fft(arr)

out1 = np.convolve(arr,kern, mode='same')
out2 = arr_fft*kern_fft
out2 = np.fft.ifft(out2)
# print(kern)
# print(arr)
print(out1)
# print(kern_fft)
# print(arr_fft)
print(np.abs(out2))