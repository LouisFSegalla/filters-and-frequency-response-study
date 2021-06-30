# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:55:31 2021

@author: luisf
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fig, axs = plt.subplots(2,2)
fig.tight_layout()

plt.sca(axs[0][0])

img = Image.open('racoon.jpg').convert('L')
img = np.asarray(img.getdata()).reshape(img.size)
print(img.shape)
plt.imshow(img, cmap="gray")
plt.title('Imagem Original')



plt.sca(axs[0][1])
fft_image = np.fft.fft2(img)
fft_image = np.fft.fftshift(fft_image)
plt.imshow(np.abs(fft_image),interpolation='nearest')
plt.title('Imagem depois da FFT')

ncols, nrows = img.shape[0], img.shape[1]
sigmax, sigmay = 10, 10
cy, cx = nrows/2, ncols/2
x = np.linspace(0, nrows, nrows)
y = np.linspace(0, ncols, ncols)
X, Y = np.meshgrid(x, y)
#gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
# gmask = np.ones((ncols,nrows))
# gmask = np.ones((ncols, nrows))*(1.0/18.0)
gmask = np.zeros((ncols, nrows))
gmask[int((ncols-1)/2),:] = np.ones(nrows) / img.shape[0]
print(gmask.shape)

plt.sca(axs[1][1])
plt.title('Filtro Gaussiano')
fft_image_filter = fft_image * gmask
plt.imshow(np.abs(fft_image_filter))

plt.sca(axs[1][0])
plt.title('Imagem reconstruida')
img_return = np.fft.ifft2(fft_image_filter)
plt.imshow(np.abs(img_return), cmap="gray")