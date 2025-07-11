import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from tqdm import tqdm

from dataset import DeblurDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Add Restormer to path
sys.path.append('./Restormer')
from Restormer.basicsr.models.archs.restormer_arch import Restormer


def precompute(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config['output_dir'], exist_ok=True)

    print(f"Saving teacher outputs to: {config['output_dir']}")
    print("Loading teacher model (Restormer)...")

    teacher_model = Restormer().to(DEVICE)
    teacher_model.load_state_dict(torch.load(config['teacher_weights'])['params'])
    teacher_model.eval()

    print("Building dataset and dataloader...")
    dataset = DeblurDataset(blur_dir=config['blur_dir'], sharp_dir=config['sharp_dir'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    print(f"Found {len(dataset)} images to process.")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    running_ssim = 0.0
    total_batches = 0

    with torch.no_grad():
        for i, (blurry_batch, sharp_batch) in enumerate(loader):
            blurry_batch = blurry_batch.to(DEVICE)
            sharp_batch = sharp_batch.to(DEVICE)

            teacher_output = teacher_model(blurry_batch)
            teacher_output = torch.clamp(teacher_output, 0, 1)

            # Per-batch SSIM
            batch_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
            batch_ssim_score = batch_ssim(teacher_output, sharp_batch).item()

            running_ssim += batch_ssim_score
            total_batches += 1
            avg_ssim = running_ssim / total_batches

            print(f"[Batch {i+1}] SSIM: {batch_ssim_score:.4f} | Running Avg SSIM: {avg_ssim:.4f}")

            for j in range(teacher_output.size(0)):
                img_index = i * config['batch_size'] + j
                if img_index < len(dataset.image_files):
                    filename = os.path.splitext(dataset.image_files[img_index])[0] + ".pt"
                    torch.save(teacher_output[j].cpu(), os.path.join(config['output_dir'], filename))

    avg_ssim = ssim_metric.compute().item()
    print(f"\n✅ Pre-computation finished successfully.")
    print(f"📊 Teacher Output Avg SSIM vs Ground Truth: **{avg_ssim:.4f}**")



if __name__ == "__main__":
    config = {
        'blur_dir': 'data/blur/train_small',
        'sharp_dir': 'data/sharp/train_small',
        'output_dir': 'data/teacher_outputs/train_small',  # Save teacher outputs as .pt files
        'teacher_weights': 'Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth',
        'batch_size': 8
    }
    precompute(config)

