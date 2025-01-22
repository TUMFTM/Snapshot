# imports
import sys
import yaml
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader

# local imports
sys.path.append(".")
from utils.evaluation import calculate_ADE_FDE, compute_pos
from model.snapshot.dataloader import SSDataset
from model.snapshot.snapshot import Snapshot

# global variables
with open(Path(__file__).parent.resolve() / "../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)


def test_model(weights_path: str, logger: logging.Logger):
    logger.info("Starting Testing the Model...")

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model and dataset
    test_dataset = SSDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=_CONFIG["pytorch"]["BATCH_SIZE"], shuffle=False, num_workers=_CONFIG["pytorch"]["NUM_WORKERS"])

    model = Snapshot().to(device)
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)

    model.eval()
    batch_iterator = tqdm(test_loader, desc="Testing the Model", colour='blue')
    with torch.no_grad():
        total_test_ADE = torch.zeros(1).to(device)
        total_test_FDE = torch.zeros(1).to(device)
        for test_data in batch_iterator:
            map_matrix, social_matrix, ground_truth = test_data
            map_matrix, social_matrix, ground_truth = map_matrix.to(device), social_matrix.to(device), ground_truth.to(device)
    
            # only two timesteps and closest 100 segments
            map_matrix = map_matrix[:, :100]

            # limit observed timesteps
            # social_matrix[:, :, 5:] = 0.0

            outputs = model(map_matrix, social_matrix) # N x 60 x 2
            pred_pos = compute_pos(outputs)
            test_ADE, test_FDE = calculate_ADE_FDE(pred_pos, ground_truth)

            total_test_ADE += test_ADE
            total_test_FDE += test_FDE

    torch.cuda.empty_cache()
    average_test_ADE = total_test_ADE / len(batch_iterator)
    average_test_FDE = total_test_FDE / len(batch_iterator)
    logger.info(f"Test ADE: {average_test_ADE.item():.4f}, Test FDE: {average_test_FDE.item():.4f}")
    logger.info("Testing finished. \n")


if __name__ == "__main__":

    # Configure logging
    date = datetime.now().strftime("%d%m%Y-%H%M%S")
    _TRAINING_PATH = Path(_CONFIG["path"]["TRAINING"]) / "snapshot" / date
    _TRAINING_PATH.mkdir(parents=True, exist_ok=True)
    log_filepath = _TRAINING_PATH / "Log_Test_Program.txt"

    _LOGGER_TEST = logging.getLogger('Logger_TEST')
    _LOGGER_TEST.setLevel(logging.DEBUG)  # Definiere das Logging-Level

    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_formatter = logging.Formatter('%(asctime)s @Test_Program %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('@Test_Program %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    _LOGGER_TEST.addHandler(file_handler)
    _LOGGER_TEST.addHandler(console_handler)

    parser = argparse.ArgumentParser(description="Tool for testing the model.")
    parser.add_argument('path', type=str, help='Path to the model weights.')
    args = parser.parse_args()
    test_model(args.path, _LOGGER_TEST)