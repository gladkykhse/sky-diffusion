import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GIFDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".gif")])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        gif_path = os.path.join(self.folder_path, self.file_list[idx])
        gif = Image.open(gif_path)

        frames = []
        for i in range(gif.n_frames):
            gif.seek(i)
            frame = gif.convert("RGB")

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        return torch.stack(frames, dim=1)
