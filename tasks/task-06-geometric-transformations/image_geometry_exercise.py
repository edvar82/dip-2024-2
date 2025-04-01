# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    """
    Apply geometric transformations to a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image as a 2D NumPy array.
        
    Returns:
        dict: Dictionary containing transformed images
    """
    
    # Ensure the image is 2D
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D array")
    
    height, width = img.shape
    
    # 1. Translation (shift right by 50px and down by 30px)
    shift_x, shift_y = 50, 30
    translated = np.zeros_like(img)
    translated[shift_y:, shift_x:] = img[:height-shift_y, :width-shift_x]
    
    # 2. Rotation (90 degrees clockwise)
    rotated = np.flip(img.T, axis=1)
    
    # 3. Horizontal stretch (scale width by 1.5)
    new_width = int(width * 1.5)
    stretched = np.zeros((height, new_width), dtype=img.dtype)
    
    # Create coordinate maps for faster interpolation
    x_indices = np.arange(new_width)
    x_orig = x_indices / 1.5
    x_floor = np.floor(x_orig).astype(int)
    x_ceil = np.minimum(x_floor + 1, width - 1)
    alpha = x_orig - x_floor

    for y in range(height):
        stretched[y] = (1 - alpha) * img[y, x_floor] + alpha * img[y, x_ceil]
    
    # 4. Horizontal mirror (flip along vertical axis)
    mirrored = np.fliplr(img)
    
    # 5. Barrel distortion
    distorted = np.zeros_like(img)
    center_y, center_x = height // 2, width // 2
    
    k = 0.00003 
    
    for y in range(height):
        for x in range(width):
            norm_y = 2.0 * (y - center_y) / height
            norm_x = 2.0 * (x - center_x) / width
            
            r = norm_x**2 + norm_y**2
            
            distortion = 1.0 + k * r
            
            src_y = center_y + (y - center_y) * distortion
            src_x = center_x + (x - center_x) * distortion
            
            if (0 <= src_y < height - 1) and (0 <= src_x < width - 1):
                src_y_floor, src_x_floor = int(src_y), int(src_x)
                src_y_ceil = min(src_y_floor + 1, height - 1)
                src_x_ceil = min(src_x_floor + 1, width - 1)
                
                y_alpha = src_y - src_y_floor
                x_alpha = src_x - src_x_floor
                
                top_left = img[src_y_floor, src_x_floor]
                top_right = img[src_y_floor, src_x_ceil]
                bottom_left = img[src_y_ceil, src_x_floor]
                bottom_right = img[src_y_ceil, src_x_ceil]
                
                top = (1 - x_alpha) * top_left + x_alpha * top_right
                bottom = (1 - x_alpha) * bottom_left + x_alpha * bottom_right
                
                distorted[y, x] = (1 - y_alpha) * top + y_alpha * bottom
    
    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
