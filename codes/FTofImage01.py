#%%
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

Original_Image = imread(os.path.join("photos\\cat2.jpg"))
Gray_Image = np.mean(Original_Image, -1); # Convert RGB to grayscale

FIgure,axs = plt.subplots(1,3)

# Plot image
Image = axs[0].imshow(Gray_Image)
Image.set_cmap('gray')
axs[0].axis('off')

# Compute row-wise FFT

Cshift = np.zeros_like(Gray_Image,dtype='complex_')
C = np.zeros_like(Gray_Image,dtype='complex_')

for j in range(Gray_Image.shape[0]):
    Cshift[j,:] = np.fft.fftshift(np.fft.fft(Gray_Image[j,:]))
    C[j,:] = np.fft.fft(Gray_Image[j,:])
    
Image = axs[1].imshow(np.log(np.abs(Cshift)))
Image.set_cmap('gray')
axs[1].axis('off')

# Compute column-wise FFT

D = np.zeros_like(C)
for j in range(C.shape[1]):
    D[:,j] = np.fft.fft(C[:,j])

Image = axs[2].imshow(np.fft.fftshift(np.log(np.abs(D))))
Image.set_cmap('gray')
axs[2].axis('off')

plt.show()

# Much more efficient to use fft2
D = np.fft.fft2(Gray_Image)
# %%
