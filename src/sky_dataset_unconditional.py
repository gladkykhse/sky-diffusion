import os
from PIL import Image
from torch.utils.data import Dataset


class SkyDatasetUnconditional(Dataset):
    def __init__(self, root_data_folder="/projects/SkyGAN/clouds_fisheye", desc_file="processed_1K_JPGs.txt",
                 transform=None):
        self._root_data_folder = root_data_folder
        self._desc_file = desc_file

        self._image_path_list = self._get_image_paths()
        self._n_samples = len(self._image_path_list)

        self._transform = transform

    def _get_image_paths(self):
        file_path = os.path.join(self._root_data_folder, self._desc_file)
        try:
            with open(file_path) as f:
                paths = f.read().strip().split('\n')
                paths = list(map(lambda x: os.path.join(self._root_data_folder, x), paths))
            return paths
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {self._desc_file} enumerating all image paths")
        except IOError:
            raise IOError(f"An IOError occured while reading file {file_path}. Check correctness of the contents")

    def __getitem__(self, item):
        try:
            sample = Image.open(self._image_path_list[item])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not open file {self._image_path_list[item]}. . Check correctness of the description file {self._desc_file}")

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return len(self._image_path_list)
