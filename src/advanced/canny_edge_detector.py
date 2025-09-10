"""
Canny edge detection algorithm.

This module provides functions for implementing the Canny edge detection algorithm.
"""

import numpy as np
import math
from ..filters.convolution import my_conv2d
from ..filters.kernel_generators import create_gaussian_kernel, create_sobel_x_kernel, create_sobel_y_kernel
from ..filters.blurring import gaussian_blur


def canny_edge_detector(image, low_threshold, high_threshold, sigma=1.0):
    """
    Apply Canny edge detection to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    low_threshold (float): Low threshold for hysteresis.
    high_threshold (float): High threshold for hysteresis.
    sigma (float): Standard deviation for Gaussian blur. Default is 1.0.

    Returns:
    numpy.ndarray: Edge-detected image.
    """
    # Step 1: Gaussian Blur
    blurred = gaussian_blur(image, size=5, sigma=sigma)

    # Step 2: Find Intensity Gradient
    sobel_x = create_sobel_x_kernel()
    sobel_y = create_sobel_y_kernel()

    grad_x = my_conv2d(blurred, sobel_x)
    grad_y = my_conv2d(blurred, sobel_y)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_direction = np.arctan2(grad_y, grad_x) * 180 / math.pi

    # Step 3: Non-Maximum Suppression
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Step 4: Hysteresis Thresholding
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)

    return edges.astype(np.uint8)


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """
    Perform non-maximum suppression on the gradient magnitude.

    Parameters:
    gradient_magnitude (numpy.ndarray): Gradient magnitude.
    gradient_direction (numpy.ndarray): Gradient direction in degrees.

    Returns:
    numpy.ndarray: Suppressed gradient magnitude.
    """
    height, width = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = gradient_direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            else:
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]

            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed


def hysteresis_thresholding(image, low_threshold, high_threshold):
    """
    Perform hysteresis thresholding.

    Parameters:
    image (numpy.ndarray): Input image after non-maximum suppression.
    low_threshold (float): Low threshold.
    high_threshold (float): High threshold.

    Returns:
    numpy.ndarray: Binary edge image.
    """
    height, width = image.shape
    edges = np.zeros_like(image, dtype=np.uint8)

    # Strong edges
    strong_edges = image >= high_threshold
    # Weak edges
    weak_edges = (image >= low_threshold) & (image < high_threshold)

    edges[strong_edges] = 255

    # Connect weak edges to strong edges
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if weak_edges[i, j]:
                if np.any(edges[i - 1:i + 2, j - 1:j + 2] == 255):
                    edges[i, j] = 255

    return edges
