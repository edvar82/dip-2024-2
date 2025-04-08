# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import numpy as np

"""
Explicação:
A função percorre cada canal (R, G e B) separadamente. Para cada canal, calcula-se o histograma e a função de distribuição cumulativa (CDF) no source e no reference. Depois, mapeia-se a CDF do source para a do reference via interpolação, ajustando assim a distribuição de intensidade de cada canal. Por fim, junta-se tudo em uma imagem resultante, que fica com a aparência cromática semelhante à imagem de referência.
"""


def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img)
    for channel in range(3):
        src_channel = source_img[:, :, channel]
        ref_channel = reference_img[:, :, channel]
        src_hist, bins = np.histogram(src_channel.ravel(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_channel.ravel(), 256, [0, 256])
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)
        src_cdf_normalized = src_cdf / src_cdf[-1]
        ref_cdf_normalized = ref_cdf / ref_cdf[-1]
        interp_map = np.zeros(256)
        j = 0
        for i in range(256):
            while j < 255 and ref_cdf_normalized[j] < src_cdf_normalized[i]:
                j += 1
            if j > 0 and j < 255:
                f = (src_cdf_normalized[i] - ref_cdf_normalized[j-1]) / (ref_cdf_normalized[j] - ref_cdf_normalized[j-1])
                interp_map[i] = (j-1) + f
            else:
                interp_map[i] = j
        matched_img[:, :, channel] = interp_map[src_channel]
    matched_img = np.clip(matched_img, 0, 255)
    return matched_img.astype(np.uint8)

