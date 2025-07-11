import os
import cv2
import numpy as np


INPUT_FOLDER = 'data/initial'
SHARP_OUTPUT_FOLDER = 'data/sharp'
BLUR_OUTPUT_FOLDER = 'data/blur'
TARGET_SIZE = (256, 256) # The uniform output size (height, width)

def resize_and_pad(image, target_size):
    """
    Resizes an image to a target size, preserving aspect ratio by padding.
    This prevents image distortion.

    Args:
        image (np.array): The input image loaded via OpenCV.
        target_size (tuple): The desired output size as (height, width).

    Returns:
        np.array: The resized and padded image.
    """
    h, w, _ = image.shape
    target_h, target_w = target_size

    # Determine the scale factor and new dimensions
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize using INTER_AREA, which is best for shrinking images
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of the target size
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calculate padding to center the image
    top_pad = (target_h - new_h) // 2
    left_pad = (target_w - new_w) // 2

    # Paste the resized image onto the center of the canvas
    padded_image[top_pad:top_pad + new_h, left_pad:left_pad + new_w] = resized_image

    return padded_image

def apply_video_call_blur(image):
    """
    Applies a Gaussian blur to simulate the effect seen in video calls.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The blurred image.
    """
    # A 15x15 kernel with a sigma of 5 provides a moderate, realistic blur.
    # You can adjust these values to change the blur intensity.
    blurred_image = cv2.GaussianBlur(image, (15, 15), sigmaX=5, sigmaY=5)
    return blurred_image

def process_images():
    """
    Main function to orchestrate the image processing pipeline.
    """
    # Create the output directories if they don't already exist
    os.makedirs(SHARP_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(BLUR_OUTPUT_FOLDER, exist_ok=True)

    # Get a sorted list of image files to ensure consistent ordering
    try:
        image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"Error: The input folder '{INPUT_FOLDER}' was not found.")
        print("Please make sure your images are in the 'data/initial' directory before running.")
        return

    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(image_files)} images. Starting processing...")

    # Loop through all found images, process, and save them
    for idx, filename in enumerate(image_files, start=1):
        # Construct the full path to the input image
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)

        # Skip if the image can't be read
        if img is None:
            print(f"Warning: Could not read image '{filename}'. Skipping.")
            continue

        # 1. Resize and pad the image to create the 'sharp' version
        sharp_img = resize_and_pad(img, TARGET_SIZE)

        # 2. Apply blur to the sharp image to create the 'blur' version
        blurred_img = apply_video_call_blur(sharp_img)

        # Define the sequential output filename. Using .png is recommended for lossless quality.
        output_filename = f'image_{idx:04d}.png'

        # Save the sharp and blurred images to their respective folders
        sharp_path = os.path.join(SHARP_OUTPUT_FOLDER, output_filename)
        blur_path = os.path.join(BLUR_OUTPUT_FOLDER, output_filename)
        
        cv2.imwrite(sharp_path, sharp_img)
        cv2.imwrite(blur_path, blurred_img)

    print(f"\nProcessing complete.")
    print(f" - {len(image_files)} sharp images saved to '{SHARP_OUTPUT_FOLDER}'")
    print(f" - {len(image_files)} blurred images saved to '{BLUR_OUTPUT_FOLDER}'")


# --- Run the main function ---
if __name__ == "__main__":
    process_images()
