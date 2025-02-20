import numpy as np

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """
    
    ### START CODE HERE ###
    width = image.shape[1]
    height = image.shape[0] 
    dtype = image.dtype
    depth = image.ndim
    min_val = np.min(image)
    max_val = np.max(image)
    mean_val = np.mean(image)
    std_val = np.std(image)
    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")