# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1, i2):

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def calculate_mse(i1, i2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((i1 - i2) ** 2)

def calculate_psnr(mse, max_pixel=1.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

def calculate_ssim(i1, i2):
    """Calculate a simplified version of Structural Similarity Index."""
    # Constants to avoid instability
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    # Calculate means
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return numerator / denominator

def calculate_npcc(i1, i2):
    """Calculate Normalized Pearson Correlation Coefficient."""
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    
    numerator = np.sum((i1 - mu1) * (i2 - mu2))
    denominator = np.sqrt(np.sum((i1 - mu1)**2) * np.sum((i2 - mu2)**2))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    """
    Compare two images and return a dictionary with the metrics:
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Structural Similarity Index (SSIM)
    - Normalized Pearson Correlation Coefficient (NPCC)
    """
    mse = calculate_mse(i1, i2)
    psnr = calculate_psnr(mse)
    ssim = calculate_ssim(i1, i2)
    npcc = calculate_npcc(i1, i2)
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "npcc": float(npcc)
    }

