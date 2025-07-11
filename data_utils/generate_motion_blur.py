import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

def apply_motion_blur(img, degree=15, angle=45):
    k = np.zeros((degree, degree))
    k[int((degree - 1) / 2), :] = np.ones(degree)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0), (degree, degree))
    k = k / degree
    return cv2.filter2D(img, -1, k)

def process_split(split_name, sharp_root='data/sharp', blur_root='data/motion_blur'):
    input_dir = os.path.join(sharp_root, split_name)
    output_dir = os.path.join(blur_root, split_name)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(os.path.join(input_dir, '*'))
    print(f"\nProcessing split: {split_name} ({len(image_paths)} images)")

    for path in tqdm(image_paths, desc=f"Motion blur - {split_name}"):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        degree = np.random.randint(5, 30)
        angle = np.random.randint(0, 360)

        blurred = apply_motion_blur(img, degree, angle)
        filename = os.path.basename(path)
        Image.fromarray(blurred).save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    for split in ['train_small', 'val_small']:
        process_split(split)
