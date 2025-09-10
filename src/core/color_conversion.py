"""
Color space conversion functions.

This module provides functions for converting between different color spaces.
"""

import numpy as np


def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using the weighted average formula.

    Parameters:
    image (numpy.ndarray): RGB image with shape (height, width, 3).

    Returns:
    numpy.ndarray: Grayscale image with shape (height, width).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB with shape (height, width, 3)")

    # Weighted average: 0.299*R + 0.587*G + 0.114*B
    grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return grayscale.astype(np.uint8)


def rgb_to_hsv(image):
    """
    Convert an RGB image to HSV color space.

    Parameters:
    image (numpy.ndarray): RGB image with shape (height, width, 3).

    Returns:
    numpy.ndarray: HSV image with shape (height, width, 3).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB with shape (height, width, 3)")

    # Normalize RGB values to [0, 1]
    r, g, b = image[:, :, 0] / 255.0, image[:, :, 1] / 255.0, image[:, :, 2] / 255.0

    # Compute Value (V)
    v = np.max([r, g, b], axis=0)

    # Compute Saturation (S)
    min_rgb = np.min([r, g, b], axis=0)
    s = np.where(v != 0, (v - min_rgb) / v, 0)

    # Compute Hue (H)
    delta = v - min_rgb
    h = np.zeros_like(v)
    h = np.where(delta != 0, np.where(v == r, (g - b) / delta, h), h)
    h = np.where(delta != 0, np.where(v == g, 2 + (b - r) / delta, h), h)
    h = np.where(delta != 0, np.where(v == b, 4 + (r - g) / delta, h), h)
    h = (h / 6) % 1  # Normalize to [0, 1]

    # Stack H, S, V into HSV image
    hsv = np.stack([h, s, v], axis=2)
    return hsv


def hsv_to_rgb(image):
    """
    Convert an HSV image to RGB color space.

    Parameters:
    image (numpy.ndarray): HSV image with shape (height, width, 3).

    Returns:
    numpy.ndarray: RGB image with shape (height, width, 3).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be HSV with shape (height, width, 3)")

    h, s, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert H from [0, 1] to [0, 360]
    h = h * 360

    # Compute RGB
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    # Assign RGB based on hue sector
    mask = (h >= 0) & (h < 60)
    r[mask] = c[mask]
    g[mask] = x[mask]

    mask = (h >= 60) & (h < 120)
    r[mask] = x[mask]
    g[mask] = c[mask]

    mask = (h >= 120) & (h < 180)
    g[mask] = c[mask]
    b[mask] = x[mask]

    mask = (h >= 180) & (h < 240)
    g[mask] = x[mask]
    b[mask] = c[mask]

    mask = (h >= 240) & (h < 300)
    r[mask] = x[mask]
    b[mask] = c[mask]

    mask = (h >= 300) & (h < 360)
    r[mask] = c[mask]
    b[mask] = x[mask]

    r = (r + m) * 255
    g = (g + m) * 255
    b = (b + m) * 255

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    return rgb
