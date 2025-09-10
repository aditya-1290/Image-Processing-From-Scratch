"""
Edge detection functions.

This module provides functions for detecting edges in images using various operators.
"""

import numpy as np
import math
from .convolution import my_conv2d
from .kernel_generators import create_sobel_x_kernel, create_sobel_y_kernel, create_prewitt_x_kernel, create_prewitt_y_kernel, create_scharr_x_kernel, create_scharr_y_kernel


def sobel_filter(image):
    """
    Apply Sobel edge detection to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).

    Returns:
    numpy.ndarray: Edge-detected image.
    """
    sobel_x = create_sobel_x_kernel()
    sobel_y = create_sobel_y_kernel()

    grad_x = my_conv2d(image, sobel_x)
    grad_y = my_conv2d(image, sobel_y)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude


def prewitt_filter(image):
    """
    Apply Prewitt edge detection to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).

    Returns:
    numpy.ndarray: Edge-detected image.
    """
    prewitt_x = create_prewitt_x_kernel()
    prewitt_y = create_prewitt_y_kernel()

    grad_x = my_conv2d(image, prewitt_x)
    grad_y = my_conv2d(image, prewitt_y)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude


def scharr_filter(image):
    """
    Apply Scharr edge detection to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).

    Returns:
    numpy.ndarray: Edge-detected image.
    """
    scharr_x = create_scharr_x_kernel()
    scharr_y = create_scharr_y_kernel()

    grad_x = my_conv2d(image, scharr_x)
    grad_y = my_conv2d(image, scharr_y)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude
