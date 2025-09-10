"""
Convolution functions.

This module provides functions for performing 2D convolution on images.
"""

import numpy as np


def my_conv2d(image, kernel, padding='zero'):
    """
    Perform 2D convolution on an image with a given kernel.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    kernel (numpy.ndarray): 2D convolution kernel.
    padding (str): Padding mode. Options: 'zero', 'replicate'. Default is 'zero'.

    Returns:
    numpy.ndarray: Convolved image.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D (grayscale)")
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Apply padding
    if padding == 'zero':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    elif padding == 'replicate':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    else:
        raise ValueError("Invalid padding mode. Choose 'zero' or 'replicate'")

    # Initialize output image
    output = np.zeros_like(image, dtype=np.float32)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output.astype(np.uint8)
