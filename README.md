# The Fundamental Image Processing Library (From First Principles)

## Project Description & Objective

The goal of this project is to build a complete, from-scratch image processing library in Python, using only the foundational numerical library NumPy for array operations and Matplotlib for visualization. The strict constraint is to avoid all high-level vision libraries like OpenCV, Scikit-Image, and Pillow's built-in functions for core algorithms.

You will implement every mathematical operation, filter kernel, and algorithm by hand. This project will demystify the "black box" of image processing and provide a deep, intuitive understanding of how pixels are manipulated to create powerful effects and extract information.

**Core Value Proposition:** To understand not just how to use an image processing function, but why it works, its mathematical basis, and how to implement it from the ground up.

## The "No-Library" Rule & Allowed Tools

### Allowed:
- **numpy:** Only for creating arrays, basic array slicing, and mathematical operations (e.g., np.array, np.zeros, np.sum, np.exp, *, +, -).
- **matplotlib:** Solely for reading images (plt.imread), displaying arrays as images (plt.imshow), and plotting graphs.
- **math:** For basic mathematical functions like sqrt, atan2, pi.

### Forbidden:
- cv2.* (OpenCV)
- skimage.* (Scikit-Image)
- PIL.Image.* functions beyond opening/saving (e.g., PIL.Image.filter is forbidden).
- scipy.ndimage.*
- numpy.convolve or numpy.correlate (You must write your own convolution function.)
- Any other library that implements vision algorithms for you.

## Implementation Modules & Algorithms

Your library will be built in modules, progressing from basic to advanced.

### Module 1: Image Fundamentals & Basic Manipulation
- **Image Loading & Display:** Use plt.imread() and plt.imshow().
- **Color Space Conversion:**
  - RGB to Grayscale: Implement the weighted average: 0.299*R + 0.587*G + 0.114*B.
  - RGB to HSV: Write functions to convert each pixel to Hue, Saturation, Value using geometric formulas.
- **Basic Point Operations:** Brightness adjustment, contrast stretching, inversion using element-wise array math.

### Module 2: Image Filtering & Convolution (The Heart of the Project)
- **2D Convolution Function:** Write a function my_conv2d(image, kernel) that:
  - Accepts a 2D grayscale image and a 2D kernel (e.g., 3x3 blur kernel).
  - Handles borders (e.g., by zero-padding or replicating the edge pixels).
  - Slides the kernel across the image, computing the sum of element-wise multiplications at each location.
- **Kernel Creation:**
  - Box Blur: Create a kernel of all 1s and normalize it.
  - Gaussian Blur: Write a function to generate a 2D Gaussian kernel from the Gaussian formula G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2)).
- **Edge Detection Kernels:**
  - Sobel Filter: Create and apply the Sobel_x ([[-1,0,1],[-2,0,2],[-1,0,1]]) and Sobel_y kernels. Compute the gradient magnitude: sqrt(sobel_x^2 + sobel_y^2).

### Module 3: Morphological Operations (Binary Image Processing)
- **Erosion:** Scan a kernel. The output pixel is 1 only if all pixels under the kernel are 1.
- **Dilation:** The output pixel is 1 if any pixel under the kernel is 1.
- **Opening (Erosion followed by Dilation):** To remove noise.
- **Closing (Dilation followed by Erosion):** To close holes.

### Module 4: Advanced Algorithms from Scratch
- **Otsu's Binarization:**
  - Compute the image histogram.
  - Iterate over all possible threshold values.
  - Calculate the within-class variance for each threshold.
  - Choose the threshold that minimizes this variance.
- **Canny Edge Detector (The Capstone):**
  - Apply Gaussian Blur.
  - Find Intensity Gradient: Use your Sobel filters.
  - Non-Maximum Suppression: Thin the edges by checking if a pixel is a local maximum along the gradient direction. This requires careful interpolation.
  - Hysteresis Thresholding: Use a high and low threshold. Strong edges are kept. Weak edges are only kept if they are connected to strong edges.

### Module 5: Feature Detection & Description (Bonus Challenge)
- **Harris Corner Detection:**
  - Compute image derivatives Ix, Iy (using your Sobel filters).
  - For each pixel, create the structure tensor M (a 2x2 matrix involving sums of Ix², Iy², Ix*Iy over a window).
  - Compute the corner response function R = det(M) - k*(trace(M))².
  - Find local maxima of R above a threshold.

## Project Structure

```
image_processing_from_scratch/
│
├── README.md
│   # Project philosophy, list of implemented functions, "no-library" rule
│
├── requirements.txt
│   # numpy, matplotlib
│
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── image_io.py          # Wrapper for plt.imread/imshow
│   │   ├── color_conversion.py  # rgb_to_grayscale, rgb_to_hsv
│   │   └── point_operations.py  # adjust_brightness, adjust_contrast, invert
│   │
│   ├── filters/
│   │   ├── convolution.py       # my_conv2d function, handle padding
│   │   ├── kernel_generators.py # create_gaussian_kernel, create_box_kernel
│   │   ├── blurring.py          # gaussian_blur, median_blur (implement a sorting-based median filter)
│   │   └── edge_detection.py    # sobel_filter, prewitt_filter, scharr_filter
│   │
│   ├── morphology/
│   │   ├── binary_ops.py        # erosion, dilation, opening, closing
│   │   └── structuring_elements.py # create circular, rectangular kernels
│   │
│   └── advanced/
│       ├── segmentation/
│       │   ├── otsu_thresholding.py
│       │   └── watershed.py     # (Very advanced from scratch)
│       ├── canny_edge_detector.py # The full pipeline
│       └── harris_corner_detector.py
│
└── examples/
    ├── 01_basic_manipulation.ipynb
    ├── 02_filtering_showcase.ipynb
    ├── 03_canny_edge_detection.ipynb
    └── 04_harris_corners.ipynb
    # Jupyter notebooks to demonstrate each module's functionality
```

## Validation & Testing

How do you know your code is correct?

- **Visual Comparison:** Process a simple image (e.g., a square) with your function and OpenCV's function. The results should be visually identical.
- **Numerical Comparison:** On a small, custom array (e.g., 5x5), calculate the expected output of a filter by hand and ensure your function matches it.
- **Benchmark Images:** Use standard images like "lena" or "cameraman" to compare your outputs with known good results.

## Learning Outcomes

Upon completion, you will have an unparalleled understanding of:

- The precise mathematical operations behind every common image filter.
- How to handle computational challenges like memory management and border effects.
- The inner workings of complex algorithms like Canny edge detection, which are often treated as a single function call.
- The ability to implement, debug, and optimize low-level computer vision algorithms from a research paper.

This project is not for the faint of heart, but it is arguably the single most effective way to transition from someone who uses computer vision to someone who truly understands it.
