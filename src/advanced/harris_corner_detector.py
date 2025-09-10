"""
Harris corner detection algorithm.

This module provides functions for implementing the Harris corner detection algorithm.
"""

import numpy as np
from ..filters.convolution import my_conv2d
from ..filters.kernel_generators import create_sobel_x_kernel, create_sobel_y_kernel


def harris_corner_detector(image, k=0.04, threshold=0.01, window_size=3):
    """
    Detect corners in an image using the Harris corner detection algorithm.

    Parameters:
    image (numpy.ndarray): Input 2D image (grayscale).
    k (float): Harris corner response parameter. Default is 0.04.
    threshold (float): Threshold for corner response. Default is 0.01.
    window_size (int): Size of the window for computing the structure tensor. Default is 3.

    Returns:
    numpy.ndarray: Corner response map.
    """
    # Compute image derivatives
    sobel_x = create_sobel_x_kernel()
    sobel_y = create_sobel_y_kernel()

    Ix = my_conv2d(image, sobel_x).astype(np.float32)
    Iy = my_conv2d(image, sobel_y).astype(np.float32)

    # Compute products of derivatives
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # Compute sums over window
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)

    Sx2 = my_conv2d(Ix2, kernel)
    Sy2 = my_conv2d(Iy2, kernel)
    Sxy = my_conv2d(Ixy, kernel)

    # Compute corner response function
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    R = det_M - k * (trace_M ** 2)

    # Threshold the response
    corners = np.zeros_like(R)
    corners[R > threshold * R.max()] = 255

    return corners.astype(np.uint8)


def find_corner_points(corner_response, min_distance=10):
    """
    Find corner points from the corner response map.

    Parameters:
    corner_response (numpy.ndarray): Corner response map.
    min_distance (int): Minimum distance between corners. Default is 10.

    Returns:
    list: List of (x, y) coordinates of detected corners.
    """
    height, width = corner_response.shape
    corners = []

    for i in range(height):
        for j in range(width):
            if corner_response[i, j] > 0:
                # Check if it's a local maximum
                is_local_max = True
                for di in range(-min_distance // 2, min_distance // 2 + 1):
                    for dj in range(-min_distance // 2, min_distance // 2 + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if corner_response[ni, nj] > corner_response[i, j]:
                                is_local_max = False
                                break
                    if not is_local_max:
                        break
                if is_local_max:
                    corners.append((j, i))  # (x, y) format

    return corners
