"""
Image Input/Output functions.

This module provides functions for loading and displaying images using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_image(image_path):
    """
    Load an image from a file path.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: The loaded image as a NumPy array.
    """
    return mpimg.imread(image_path)


def display_image(image, title=None):
    """
    Display an image using matplotlib.

    Parameters:
    image (numpy.ndarray): The image to display.
    title (str, optional): Title for the image display.
    """
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()


def save_image(image, output_path):
    """
    Save an image to a file path.

    Parameters:
    image (numpy.ndarray): The image to save.
    output_path (str): Path to save the image.
    """
    mpimg.imsave(output_path, image)
