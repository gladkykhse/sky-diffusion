import argparse
import os

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers import UNet2DModel

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", default=10, type=int)
parser.add_argument("--image_size", default=512, type=int)
parser.add_argument("--mixed_precision", default="fp16", type=str)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1,
    type=int,
)
parser.add_argument(
    "--output_dir",
    default="unconditional_sky_diffusion",
    type=str,
)
parser.add_argument("--samples_dir", default="samples", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model_path", type=str)


def create_model(
    config: argparse.Namespace,
) -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def main(config: argparse.Namespace):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = create_model(config)
    model.load_state_dict(torch.load(config.model_path))
    model.to(device)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(
        unet=accelerator.unwrap_model(model),
        scheduler=noise_scheduler,
    )

    images = pipeline(
        batch_size=config.n_samples,
        generator=torch.manual_seed(config.seed),
    ).images

    saving_path = os.path.join(config.output_dir, config.samples_dir)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print(f"Folder {saving_path} created")

    for i, image in enumerate(images):
        image.save(f"{saving_path}/sample_{i + 1}.png")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
