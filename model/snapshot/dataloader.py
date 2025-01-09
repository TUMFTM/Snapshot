# imports
import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset

# global variables
with open("config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)


class SSDataset(Dataset):
    def __init__(self, mode):
        """
        Initializes an instance of the SSDataset class.

        Parameters:
            mode (str): The mode of the dataset (e.g., 'train', 'test', 'val').

        Global Variables:
            _CONFIG (dict): A dictionary containing the configuration parameters from the config.yaml file.

        Raises:
            RuntimeError: If no files are found in the specified directory.

        """
        self.mode = mode
        self.directory = Path(_CONFIG["path"]["PREPROCESSED_DATA"]) / _CONFIG["path"]["PREPROCESSED_DIR"] / mode
        self.files = sorted(self.directory.glob("*/"))
        if not self.files:
            raise RuntimeError("A critical error has occurred. The program will terminate.")



    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of files in the dataset.
        """
        return len(self.files)



    def __getitem__(self, idx):
        """
        Retrieves the data at the given index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the map matrix, social matrix, and ground truth data.
        """
        target_dir = self.files[idx]
        map_matrix = torch.load(target_dir / "map_matrix.pt")
        social_matrix = torch.load(target_dir / "social_matrix.pt")
        ground_truth = torch.load(target_dir / "ground_truth.pt")
        
        return map_matrix, social_matrix, ground_truth