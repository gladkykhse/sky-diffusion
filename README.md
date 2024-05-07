# Denoising Diffusion Models for Dynamic Sky Image Generation

This repository contains the code from the research conducted during my bachelor's thesis at
Charles University, Faculty of Mathematics and Physics, Prague, Czech Republic.

## Installation
A key part of this work was developed in Jupyter notebooks due to their convenience in research projects.
We used Conda as a package and environment management system for installing and managing software packages
and dependencies. You can either set up an environment using our `environment.yaml` file (for Linux machines
only), or manually install dependencies listed in `requirements.txt`.

### Setup with Conda environment file

__1. Install Conda__
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
__2. Clone project__
```shell
git clone https://github.com/gladkykhse/sky-diffusion.git && cd sky-diffusion
```
__3. Create a Conda Environment__
```shell
conda env create -f environment.yaml
```
__4. Activate the Environment__
```shell
conda activate sky_diffusion
```

### Manual setup
In this project, we utilized `Python 3.11.5`. Hence, we highly recommend using the same
version to ensure compatibility with all libraries.

You can either create a virtual environment and install all dependencies using pip:
```shell
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Or create own Conda environment and again install all dependencies:
```shell
conda create --name sky_diffusion python=3.11
conda activate sky_diffusion
pip install -r requirements.txt
```

Note: there might be a problem with `mpi4py` installation. It can be solved by installing
appropriate Open MPI library for your system (e.g. `mpich` for MacOS, or `openmpi` may help
for Linux systems)

## Project structure

- `research` - this directory contains the code utilized in this work (only final files).
  - `unconditional_generation` - Within this directory, you'll find notebooks dedicated to training, sampling, and evaluation for the unconditional image generation task using DDPM/DDIM models.
  - `conditional_generation` - This directory houses notebooks focused on training, data preparation, as well as separate notebook for sampling and evaluation for the conditional image generation task employing DDPM models.
  - `unconditional_video_generation` - Here, you'll find notebooks dedicated to training, dataset creation, sampling, and evaluation for the unconditional video generation task using VDM models.
  - `conditional_video_generation` - This directory contains notebooks dedicated to training, sampling, and evaluation for the conditional video generation task using RaMViD models.
- `data_samples` - this directory contains GIF samples extracted from the dataset that serve as conditioning data for sampling from conditional models
  - `gif_64` - 20 video samples of 64x64 resolution
  - `gif_128` - 20 video samples of 128x128 resolution


## Sampling from trained models

If you would like to use pretrained models and just generate images/videos you have to download our final models
and dataset examples from public [Google Drive](https://drive.google.com/drive/folders/1y152MTtJKnmH_0nJki5FBsSeqUuYfyn4?usp=sharing). After that you have to change the value of the variables
containing model and data paths within all sampling notebooks with the exact path to the corresponding
file/directory on your machine.