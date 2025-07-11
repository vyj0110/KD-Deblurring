import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

def apply_motion_blur(img, degree=15, angle=45):
    """
    Applies motion blur to an input image.

    Parameters:
    - img (np.ndarray): Input image in RGB format.
    - degree (int): The intensity of the blur (kernel size).
    - angle (float): The angle at which the motion blur is applied.

    Returns:
    - np.ndarray: The motion-blurred image.
    """
    # Create an empty kernel with a horizontal line in the center
    k = np.zeros((degree, degree))
    k[int((degree - 1) / 2), :] = np.ones(degree)

    # Rotate the kernel to the specified angle
    rotation_matrix = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rotation_matrix, (degree, degree))

    # Normalize the kernel
    k = k / degree

    # Apply the kernel to the image
    return cv2.filter2D(img, -1, k)

def process_split(split_name, sharp_root='data/sharp', blur_root='data/motion_blur'):
    """
    Processes all images in a split folder by applying random motion blur.

    Parameters:
    - split_name (str): The name of the data split (e.g., 'train_small', 'val_small').
    - sharp_root (str): Directory path to the original sharp images.
    - blur_root (str): Directory path where blurred images will be saved.
    """
    # Define input and output directories for the current split
    input_dir = os.path.join(sharp_root, split_name)
    output_dir = os.path.join(blur_root, split_name)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all image file paths from the input directory
    image_paths = glob(os.path.join(input_dir, '*'))
    print(f"\nProcessing split: {split_name} ({len(image_paths)} images)")

    # Apply motion blur to each image
    for path in tqdm(image_paths, desc=f"Motion blur - {split_name}"):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Randomly choose blur parameters
        degree = np.random.randint(5, 30)
        angle = np.random.randint(0, 360)

        # Apply motion blur and save the result
        blurred = apply_motion_blur(img, degree, angle)
        filename = os.path.basename(path)
        Image.fromarray(blurred).save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    # Process the specified splits
    for split in ['train_small', 'val_small']:
        process_split(split)
