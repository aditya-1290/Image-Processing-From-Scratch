"""
Binary morphological operations.

This module provides functions for erosion, dilation, opening, and closing on binary images.
"""

import numpy as np


def erosion(image, kernel):
    """
    Perform erosion on a binary image.

    Parameters:
    image (numpy.ndarray): Binary input image (0 or 1).
    kernel (numpy.ndarray): Structuring element (binary).

    Returns:
    numpy.ndarray: Eroded binary image.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            if np.all(region[kernel == 1] == 1):
                output[i, j] = 1
            else:
                output[i, j] = 0

    return output


def dilation(image, kernel):
    """
    Perform dilation on a binary image.

    Parameters:
    image (numpy.ndarray): Binary input image (0 or 1).
    kernel (numpy.ndarray): Structuring element (binary).

    Returns:
    numpy.ndarray: Dilated binary image.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            if np.any(region[kernel == 1] == 1):
                output[i, j] = 1
            else:
                output[i, j] = 0

    return output


def opening(image, kernel):
    """
    Perform opening on a binary image (erosion followed by dilation).

    Parameters:
    image (numpy.ndarray): Binary input image (0 or 1).
    kernel (numpy.ndarray): Structuring element (binary).

    Returns:
    numpy.ndarray: Image after opening.
    """
    eroded = erosion(image, kernel)
    opened = dilation(eroded, kernel)
    return opened


def closing(image, kernel):
    """
    Perform closing on a binary image (dilation followed by erosion).

    Parameters:
    image (numpy.ndarray): Binary input image (0 or 1).
    kernel (numpy.ndarray): Structuring element (binary).

    Returns:
    numpy.ndarray: Image after closing.
    """
    dilated = dilation(image, kernel)
    closed = erosion(dilated, kernel)
    return closed
