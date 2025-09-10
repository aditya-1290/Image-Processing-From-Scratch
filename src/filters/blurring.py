"""
Blurring functions.

This module provides functions for applying various blurring filters to images.
"""

import numpy as np
from .convolution import my_conv2d
from .kernel_generators import create_box_kernel, create_gaussian_kernel


def box_blur(image, size=3):
    """
    Apply box blur to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    size (int): Size of the box kernel. Default is 3.

    Returns:
    numpy.ndarray: Blurred image.
    """
    kernel = create_box_kernel(size)
    return my_conv2d(image, kernel)


def gaussian_blur(image, size=3, sigma=1.0):
    """
    Apply Gaussian blur to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    size (int): Size of the Gaussian kernel. Default is 3.
    sigma (float): Standard deviation of the Gaussian. Default is 1.0.

    Returns:
    numpy.ndarray: Blurred image.
    """
    kernel = create_gaussian_kernel(size, sigma)
    return my_conv2d(image, kernel)


def median_blur(image, size=3):
    """
    Apply median blur to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    size (int): Size of the median filter window. Default is 3.

    Returns:
    numpy.ndarray: Blurred image.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D (grayscale)")

    image_height, image_width = image.shape
    pad_size = size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + size, j:j + size]
            output[i, j] = np.median(region)

    return output.astype(np.uint8)
