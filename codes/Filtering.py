
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_pass_filter(image, cutoff_frequency):
    # Compute the 2D Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Create a frequency grid
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    
    # Create a circular low-pass filter
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_frequency:
                mask[i, j] = 1
    
    # Apply the filter to the Fourier transform
    f_transform_filtered = f_transform * mask
    
    # Inverse Fourier transform to obtain the filtered image
    filtered_image = np.fft.ifft2(f_transform_filtered).real
    return filtered_image

def high_pass_filter(image, cutoff_frequency):
    # Compute the 2D Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Create a frequency grid
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    
    # Create a circular high-pass filter
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_frequency:
                mask[i, j] = 0
    
    # Apply the filter to the Fourier transform
    f_transform_filtered = f_transform * mask
    
    # Inverse Fourier transform to obtain the filtered image
    filtered_image = np.fft.ifft2(f_transform_filtered).real
    return filtered_image

# Load the image
image = cv2.imread('images\curves.jpeg', cv2.IMREAD_GRAYSCALE)

# Set the cutoff frequency for the filters
low_pass_cutoff = 100
high_pass_cutoff = 100

# Apply the low-pass filter
filtered_image_low_pass = low_pass_filter(image, low_pass_cutoff)

# Apply the high-pass filter
filtered_image_high_pass = high_pass_filter(image, high_pass_cutoff)

# Display the original image and filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image_low_pass, cmap='gray')
plt.title('Low-Pass Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image_high_pass, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.axis('off')

plt.show()

# %%
