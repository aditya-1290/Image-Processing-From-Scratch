"""
Kernel generation functions.

This module provides functions for generating various convolution kernels.
"""

import numpy as np
import math


def create_box_kernel(size):
    """
    Create a box blur kernel.

    Parameters:
    size (int): Size of the kernel (odd number).

    Returns:
    numpy.ndarray: Box blur kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return kernel


def create_gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel.

    Parameters:
    size (int): Size of the kernel (odd number).
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    numpy.ndarray: Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel


def create_sobel_x_kernel():
    """
    Create the Sobel X kernel for edge detection.

    Returns:
    numpy.ndarray: Sobel X kernel.
    """
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    return kernel


def create_sobel_y_kernel():
    """
    Create the Sobel Y kernel for edge detection.

    Returns:
    numpy.ndarray: Sobel Y kernel.
    """
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
    return kernel


def create_prewitt_x_kernel():
    """
    Create the Prewitt X kernel for edge detection.

    Returns:
    numpy.ndarray: Prewitt X kernel.
    """
    kernel = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float32)
    return kernel


def create_prewitt_y_kernel():
    """
    Create the Prewitt Y kernel for edge detection.

    Returns:
    numpy.ndarray: Prewitt Y kernel.
    """
    kernel = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], dtype=np.float32)
    return kernel


def create_scharr_x_kernel():
    """
    Create the Scharr X kernel for edge detection.

    Returns:
    numpy.ndarray: Scharr X kernel.
    """
    kernel = np.array([[-3, 0, 3],
                       [-10, 0, 10],
                       [-3, 0, 3]], dtype=np.float32)
    return kernel


def create_scharr_y_kernel():
    """
    Create the Scharr Y kernel for edge detection.

    Returns:
    numpy.ndarray: Scharr Y kernel.
    """
    kernel = np.array([[-3, -10, -3],
                       [0, 0, 0],
                       [3, 10, 3]], dtype=np.float32)
    return kernel
