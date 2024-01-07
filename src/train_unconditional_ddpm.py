import argparse
from pathlib import Path
import os

import torch
from accelerate import Accelerator, notebook_launcher
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

from sky_dataset_unconditional import SkyDatasetUnconditional

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", default=512, required=True, type=int)
parser.add_argument("--train_batch_size", default=4, required=True, type=int)
parser.add_argument("--eval_batch_size", default=16, required=True, type=int)
parser.add_argument("--num_epochs", default=150, required=True, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, required=True, type=int)
parser.add_argument("--learning_rate", default=1e-5, required=True, type=float)
parser.add_argument("--lr_warmup_steps", default=500, required=True, type=int)
parser.add_argument("--save_image_epochs", default=5, required=True, type=int)
parser.add_argument("--save_model_epochs", default=5, required=True, type=int)
parser.add_argument("--mixed_precision", default="fp16", required=True, type=str)
parser.add_argument("--output_dir", default="../unconditional_sky_diffusion", required=True, type=str)
parser.add_argument("--seed", default=42, required=True, type=int)

# HuggingFace repo setup
parser.add_argument("--push_to_hub", default=True, required=True, type=bool)
parser.add_argument("--hub_model_id", default="../unconditional_sky_diffusion", required=True, type=str)
parser.add_argument("--hub_private_repo", default=False, required=True, type=bool)
parser.add_argument("--overwrite_output_dir", default=True, required=True, type=bool)
args = parser.parse_args()


def create_model(config: argparse.Namespace) -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
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


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    image_grid = make_image_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{(epoch + 1):04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            print()
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch + 1, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save(model.state_dict(), f"{config.output_dir}/models/model{epoch + 1}.pt")
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch + 1}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )


def main(config: argparse.Namespace):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"Folder {config.output_dir} created")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = SkyDatasetUnconditional(transform=preprocess)
    train_dataloader = DataLoader(dataset=dataset, batch_size=config.train_batch_size, shuffle=True)

    model = create_model(config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
