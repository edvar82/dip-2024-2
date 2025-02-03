import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
def generate_image(seed, width, height, mean, std):
    """
    Generates a grayscale image with pixel values sampled from a normal distribution.

    Args:
        seed (int): Random seed for reproducibility (student's registration number).
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        image (numpy.ndarray): The generated image.
    """
    ### START CODE HERE ###
    np.random.seed(seed)
    image = np.random.normal(mean, std, (height, width))
    image = np.clip(image, 0, 255) 
    ### END CODE HERE ###

    return image

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate an image with pixel values sampled from a normal distribution.")

    parser.add_argument('--registration_number', type=int, required=True, help="Student's registration number (used as seed)")
    parser.add_argument('--width', type=int, required=True, help="Width of the image")
    parser.add_argument('--height', type=int, required=True, help="Height of the image")
    parser.add_argument('--mean', type=float, required=True, help="Mean of the normal distribution")
    parser.add_argument('--std', type=float, required=True, help="Standard deviation of the normal distribution")
    parser.add_argument('--output', type=str, required=True, help="Path to save the generated image")

    args = parser.parse_args()

    # Generate the image
    image = generate_image(args.registration_number, args.width, args.height, args.mean, args.std)

    clipped_image = np.clip(image, 0, 255).astype(np.uint8) # clip the image to be between 0 and 255 to avoid overflow
    # Save the generated image
    cv2.imwrite(args.output, clipped_image)

    print(f"Image successfully generated and saved to {args.output}")

    # plot the histogram of the image
    plt.hist(image.flatten(), bins=100, density=True, alpha=0.75)
    plt.show()

if __name__ == "__main__":
    main()