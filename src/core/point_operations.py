"""
Point operations on images.

This module provides functions for basic point operations like brightness adjustment, contrast stretching, and inversion.
"""

import numpy as np


def adjust_brightness(image, delta):
    """
    Adjust the brightness of an image by adding a constant value.

    Parameters:
    image (numpy.ndarray): Input image.
    delta (int): Brightness adjustment value. Positive values increase brightness, negative values decrease it.

    Returns:
    numpy.ndarray: Brightness-adjusted image.
    """
    adjusted = image.astype(np.int32) + delta
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def adjust_contrast(image, factor):
    """
    Adjust the contrast of an image by multiplying by a factor.

    Parameters:
    image (numpy.ndarray): Input image.
    factor (float): Contrast adjustment factor. Values > 1 increase contrast, values < 1 decrease contrast.

    Returns:
    numpy.ndarray: Contrast-adjusted image.
    """
    mean = np.mean(image)
    adjusted = mean + factor * (image - mean)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def invert_image(image):
    """
    Invert the colors of an image.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Inverted image.
    """
    inverted = 255 - image
    return inverted.astype(np.uint8)


def gamma_correction(image, gamma):
    """
    Apply gamma correction to an image.

    Parameters:
    image (numpy.ndarray): Input image.
    gamma (float): Gamma value. Values < 1 make the image brighter, values > 1 make it darker.

    Returns:
    numpy.ndarray: Gamma-corrected image.
    """
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return corrected
