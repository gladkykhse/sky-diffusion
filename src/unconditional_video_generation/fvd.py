import math
import os
import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torcheval.metrics.metric import Metric


def load_i3d_pretrained(device=torch.device("cpu")):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = "i3d_torchscript.pt"
    print(filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    i3d = torch.nn.DataParallel(i3d)
    return i3d


def get_feats(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(
        rescale=False, resize=False, return_features=True
    )  # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos) - 1) // bs + 1):
            feats = np.vstack(
                [
                    feats,
                    detector(
                        torch.stack([preprocess_single(video) for video in videos[i * bs : (i + 1) * bs]]).to(device),
                        **detector_kwargs,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                ]
            )
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


class FID_I3D(nn.Module):
    def __init__(
        self,
        weights: Optional[str] = "DEFAULT",
    ) -> None:
        super().__init__()
        self.model = load_i3d_pretrained(device="cuda")

    def forward(self, x: Tensor) -> Tensor:
        vals = get_fvd_feats(x, i3d=self.model, device="cuda")
        return torch.tensor(vals)


class FrechetInceptionDistance(Metric[torch.Tensor]):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        feature_dim: int = 400,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)

        if model is None:
            model = FID_I3D()

        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self._add_state("real_sum", torch.zeros(feature_dim, device=device))
        self._add_state("real_cov_sum", torch.zeros((feature_dim, feature_dim), device=device))
        self._add_state("fake_sum", torch.zeros(feature_dim, device=device))
        self._add_state("fake_cov_sum", torch.zeros((feature_dim, feature_dim), device=device))
        self._add_state("num_real_images", torch.tensor(0, device=device).int())
        self._add_state("num_fake_images", torch.tensor(0, device=device).int())

    @torch.inference_mode()
    def update(self, images: Tensor, is_real: bool):
        images = images.to(self.device)
        activations = self.model(images)

        batch_size = images.shape[0]

        if is_real:
            self.num_real_images += batch_size
            self.real_sum += torch.sum(activations, dim=0)
            self.real_cov_sum += torch.matmul(activations.T, activations)
        else:
            self.num_fake_images += batch_size
            self.fake_sum += torch.sum(activations, dim=0)
            self.fake_cov_sum += torch.matmul(activations.T, activations)

        return self

    @torch.inference_mode()
    def compute(self) -> Tensor:
        if (self.num_real_images < 2) or (self.num_fake_images < 2):
            warnings.warn(
                "Computing FID requires at least 2 real images and 2 fake images,"
                f"but currently running with {self.num_real_images} real images and {self.num_fake_images} fake images."
                "Returning 0.0",
                RuntimeWarning,
                stacklevel=2,
            )

            return torch.tensor(0.0)

        real_mean = (self.real_sum / self.num_real_images).unsqueeze(0)
        fake_mean = (self.fake_sum / self.num_fake_images).unsqueeze(0)

        real_cov_num = self.real_cov_sum - self.num_real_images * torch.matmul(real_mean.T, real_mean)
        real_cov = real_cov_num / (self.num_real_images - 1)
        fake_cov_num = self.fake_cov_sum - self.num_fake_images * torch.matmul(fake_mean.T, fake_mean)
        fake_cov = fake_cov_num / (self.num_fake_images - 1)

        fid = self._calculate_frechet_distance(real_mean.squeeze(), real_cov, fake_mean.squeeze(), fake_cov)
        return fid

    def _calculate_frechet_distance(
        self,
        mu1: Tensor,
        sigma1: Tensor,
        mu2: Tensor,
        sigma2: Tensor,
    ) -> Tensor:
        mean_diff = mu1 - mu2
        mean_diff_squared = mean_diff.square().sum(dim=-1)

        trace_sum = sigma1.trace() + sigma2.trace()

        sigma_mm = torch.matmul(sigma1, sigma2)
        eigenvals = torch.linalg.eigvals(sigma_mm)

        sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)

        fid = mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum

        return fid

    def merge_state(self, metrics):
        for metric in metrics:
            self.real_sum += metric.real_sum.to(self.device)
            self.real_cov_sum += metric.real_cov_sum.to(self.device)
            self.fake_sum += metric.fake_sum.to(self.device)
            self.fake_cov_sum += metric.fake_cov_sum.to(self.device)
            self.num_real_images += metric.num_real_images.to(self.device)
            self.num_fake_images += metric.num_fake_images.to(self.device)

        return self

    def to(
        self,
        device: Union[str, torch.device],
        *args: Any,
        **kwargs: Any,
    ):
        super().to(device=device)
        self.model.to(self.device)
        return self
