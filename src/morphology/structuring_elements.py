"""
Structuring element generation functions.

This module provides functions for creating various structuring elements for morphological operations.
"""

import numpy as np
import math


def create_rectangular_kernel(height, width):
    """
    Create a rectangular structuring element.

    Parameters:
    height (int): Height of the kernel.
    width (int): Width of the kernel.

    Returns:
    numpy.ndarray: Rectangular structuring element.
    """
    kernel = np.ones((height, width), dtype=np.uint8)
    return kernel


def create_circular_kernel(radius):
    """
    Create a circular structuring element.

    Parameters:
    radius (int): Radius of the circle.

    Returns:
    numpy.ndarray: Circular structuring element.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = radius

    for i in range(size):
        for j in range(size):
            if math.sqrt((i - center) ** 2 + (j - center) ** 2) <= radius:
                kernel[i, j] = 1

    return kernel


def create_cross_kernel(size):
    """
    Create a cross-shaped structuring element.

    Parameters:
    size (int): Size of the cross (odd number).

    Returns:
    numpy.ndarray: Cross-shaped structuring element.
    """
    if size % 2 == 0:
        raise ValueError("Size must be odd")

    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    kernel[center, :] = 1
    kernel[:, center] = 1

    return kernel


def create_diamond_kernel(size):
    """
    Create a diamond-shaped structuring element.

    Parameters:
    size (int): Size of the diamond (odd number).

    Returns:
    numpy.ndarray: Diamond-shaped structuring element.
    """
    if size % 2 == 0:
        raise ValueError("Size must be odd")

    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2

    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) <= center:
                kernel[i, j] = 1

    return kernel
