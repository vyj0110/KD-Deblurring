import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DeblurDataset(Dataset):
    """
    to preprocess and load image
    """

    def __init__(self, blur_dir, sharp_dir, teacher_dir=None, image_size=(256, 256)):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.teacher_dir = teacher_dir
        self.image_files = [f for f in os.listdir(blur_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load blurry and sharp images
        blur_image = Image.open(os.path.join(self.blur_dir, img_name)).convert('RGB')
        sharp_image = Image.open(os.path.join(self.sharp_dir, img_name)).convert('RGB')

        blur_image = self.transform(blur_image)
        sharp_image = self.transform(sharp_image)

        if self.teacher_dir:
            pt_filename = os.path.splitext(img_name)[0] + ".pt"
            teacher_path = os.path.join(self.teacher_dir, pt_filename)
            teacher_output = torch.load(teacher_path)
            teacher_output = torch.clamp(teacher_output, 0.0, 1.0)  # Clamp for safety
            return blur_image, sharp_image, teacher_output
        else:
            return blur_image, sharp_image
