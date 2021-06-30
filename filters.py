# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:42:23 2021

@author: luisf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc

fig, axs = plt.subplots(3,1)
fig.tight_layout()

plt.sca(axs[0])

img = misc.ascent()
print(img.shape)

img_cropped = np.zeros((100,100))
print(img_cropped.shape)
for i in range(img_cropped.shape[0]):
    for j in range(img_cropped.shape[1]):
        img_cropped[i][j] = img[i][j]
        


plt.imshow(img_cropped, cmap="gray")
plt.title('Imagem Original')


img_fft   = np.fft.fft2(img_cropped)
img_mod   = np.absolute(img_fft)
img_phase = np.angle(img_fft)

plt.sca(axs[1])
plt.imshow(img_mod)
plt.title('Módulo da FFT')

plt.sca(axs[2])
plt.imshow(img_phase)
plt.title('Fase da FFT')