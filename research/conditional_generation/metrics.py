import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def compute_batch_ssim(batch1, batch2):
    ssim_values = []
    for img1, img2 in zip(batch1, batch2):
        img1 = ((img1 + 1) / 2).permute(1, 2, 0).cpu().numpy()
        img2 = ((img2 + 1) / 2).permute(1, 2, 0).cpu().numpy()

        if len(img1.shape) > 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) > 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        score, _ = ssim(img1, img2, full=True, data_range=1.0)
        ssim_values.append(score)

    avg_ssim = np.mean(ssim_values)
    return avg_ssim


def compute_batch_psnr(batch1, batch2, max_val=1.0):
    mse = torch.mean((batch1 - batch2) ** 2, dim=(1, 2, 3))
    psnr = 10 * torch.log10((max_val**2) / mse)
    avg_psnr = torch.mean(psnr)

    return avg_psnr.item()


def compute_batch_mse(batch1, batch2):
    mse = torch.mean((batch1 - batch2) ** 2, dim=(1, 2, 3))
    avg_mse = torch.mean(mse)

    return avg_mse.item()


def apply_mask(images):
    red_channel = images[:, 0, :, :]
    green_channel = images[:, 1, :, :]
    blue_channel = images[:, 2, :, :]

    condition_1 = blue_channel < (red_channel + 45)
    condition_2 = blue_channel < (green_channel + 35)
    condition_3 = red_channel > 4

    mask = condition_1 & condition_2 & condition_3
    mask = mask.float()
    return mask


def average_deviation(data):
    mean = sum(data) / len(data)
    deviations = [abs(x - mean) for x in data]
    avg_deviation = sum(deviations) / len(deviations)

    return avg_deviation
