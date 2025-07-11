import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm
from torchvision import models

from models import LightweightUNet
from dataset import DeblurDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return 1 - ssim(x, y, data_range=1.0, size_average=True)


def evaluate(model, val_loader, device):
    model.eval()
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    with torch.no_grad():
        for blurry_img, sharp_img in val_loader:
            blurry_img, sharp_img = blurry_img.to(device), sharp_img.to(device)
            output = model(blurry_img)
            psnr_metric.update(output, sharp_img)
            ssim_metric.update(output, sharp_img)
    return psnr_metric.compute(), ssim_metric.compute()


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    TRAIN_BLUR_DIR = "data/blur/train_small"
    TRAIN_SHARP_DIR = "data/sharp/train_small"
    TEACHER_OUTPUT_DIR = "data/teacher_outputs/train_small"
    VAL_BLUR_DIR = "data/blur/val_small"
    VAL_SHARP_DIR = "data/sharp/val_small"

    BATCH_SIZE = 4
    LEARNING_RATE = 5.5e-5  # Midway between 1e-4 and 1e-5
    EPOCHS = 20

    # --- Loss Weights ---
    l1_w, distill_w = 0.30, 0.7

    print("Initializing student model...")
    model = LightweightUNet().to(DEVICE)

    # --- Dataset ---
    train_dataset = DeblurDataset(TRAIN_BLUR_DIR, TRAIN_SHARP_DIR, TEACHER_OUTPUT_DIR)
    val_dataset = DeblurDataset(VAL_BLUR_DIR, VAL_SHARP_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Losses and Optimizer ---
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_ssim = 0.0

    print(f"Training for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch = 0.0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")

        for blurry, sharp, teacher in loop:
            blurry = blurry.to(DEVICE)
            sharp = sharp.to(DEVICE)
            teacher = teacher.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                output = model(blurry)
                loss_l1 = l1_loss(output, sharp)
                loss_ssim = ssim_loss(output, sharp)
                loss_distill = l1_loss(output, teacher)

                loss = (
                    l1_w * loss_l1 +
                    distill_w * loss_distill
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Skipping batch due to NaN/Inf loss.")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()
            loop.set_postfix(
                l1=loss_l1.item(),
                ssim=1 - loss_ssim.item(),
                distill=loss_distill.item(),
                total=loss.item(),
                lr=LEARNING_RATE
            )

        # Validation
        avg_loss = total_loss_epoch / len(train_loader)
        val_psnr, val_ssim = evaluate(model, val_loader, DEVICE)
        print(f"📊 Epoch {epoch+1} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | Avg Loss: {avg_loss:.4f}")

        # Save Checkpoints
        torch.save(model.state_dict(), f"student_model_epoch_{epoch+1}.pth")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), "student_model_best.pth")
            print(f"📈 New best model saved at epoch {epoch+1} with SSIM: {val_ssim:.4f}")

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
