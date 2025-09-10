"""
Otsu's thresholding algorithm.

This module provides functions for automatic thresholding using Otsu's method.
"""

import numpy as np


def compute_histogram(image):
    """
    Compute the histogram of an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).

    Returns:
    numpy.ndarray: Histogram of the image.
    """
    histogram = np.zeros(256, dtype=np.int32)
    for pixel in image.flatten():
        histogram[pixel] += 1
    return histogram


def otsu_threshold(image):
    """
    Compute the optimal threshold using Otsu's method.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).

    Returns:
    int: Optimal threshold value.
    """
    histogram = compute_histogram(image)
    total_pixels = image.size

    # Compute cumulative sums
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))

    # Compute global mean
    global_mean = cumulative_mean[-1] / total_pixels

    # Initialize variables
    max_variance = 0
    optimal_threshold = 0

    for t in range(1, 256):
        # Class probabilities
        w0 = cumulative_sum[t] / total_pixels
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        # Class means
        mu0 = cumulative_mean[t] / cumulative_sum[t]
        mu1 = (global_mean - w0 * mu0) / w1

        # Within-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t

    return optimal_threshold


def apply_threshold(image, threshold):
    """
    Apply thresholding to an image.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    threshold (int): Threshold value.

    Returns:
    numpy.ndarray: Binary image after thresholding.
    """
    binary = np.where(image >= threshold, 255, 0).astype(np.uint8)
    return binary
