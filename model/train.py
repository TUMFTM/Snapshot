# imports
import sys
import yaml
import logging
import csv
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# local imports
sys.path.append(".")
from utils.evaluation import calculate_ADE_FDE, compute_pos
from model.snapshot.dataloader import SSDataset
from model.snapshot.snapshot import Snapshot
from model.test import test_model

# global variables
with open("config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

_SPLIT = _CONFIG["pytorch"]["BATCH_SIZE"] / _CONFIG["samples"]["OBSERVATION_LENGTH"]
_RATIO = [int(_SPLIT), int(_SPLIT*2), int(_SPLIT*3), int(_SPLIT*4), int(_SPLIT*5), int(_SPLIT*6), int(_SPLIT*7), int(_SPLIT*8)]


def mask_batch(tensor):

    # batch gets newly sampled at each epoch, hence randomized timesteps selection not necessary 
    tensor[:_RATIO[0], :, 5:] = 0.0
    tensor[_RATIO[0]:_RATIO[1], :, 7:] = 0.0
    tensor[_RATIO[1]:_RATIO[2], :, 9:] = 0.0
    tensor[_RATIO[2]:_RATIO[3], :, 11:] = 0.0
    tensor[_RATIO[3]:_RATIO[4], :, 13:] = 0.0
    tensor[_RATIO[4]:_RATIO[5], :, 15:] = 0.0
    tensor[_RATIO[5]:_RATIO[6], :, 17:] = 0.0
    tensor[_RATIO[6]:_RATIO[7], :, 19:] = 0.0

    return tensor


def train_model(weight_path = None):

    _LOGGER_TRAIN.info("Starting Training the Model...")

    with open(_TRAINING_PATH/f"ADE_FDE_snapshot.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_ADE', 'Train_FDE', 'Val_ADE', 'Val_FDE'])


    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOGGER_TRAIN.info(f"Device: {device}")
    if device == "cuda":
        _LOGGER_TRAIN.info(f"Device name: {torch.cuda.get_device_name(device.index)}")
        _LOGGER_TRAIN.info(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory/ 1024 ** 3} GiB")
    device = torch.device(device)

    train_dataset = SSDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=_CONFIG["pytorch"]["BATCH_SIZE"], shuffle=True, num_workers=_CONFIG["pytorch"]["NUM_WORKERS"])

    val_dataset = SSDataset("val")
    val_loader = DataLoader(val_dataset, batch_size=_CONFIG["pytorch"]["BATCH_SIZE"], shuffle=True, num_workers=_CONFIG["pytorch"]["NUM_WORKERS"])


    model = Snapshot().to(device)
    if weight_path != None:
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights)

    # Print the model architecture and the number of parameters 
    total_params = sum(p.numel() for p in model.parameters())
    _LOGGER_TRAIN.info(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    _LOGGER_TRAIN.info(f"{total_trainable_params:,} training parameters.\n")

    optimizer = optim.Adam(model.parameters(), lr=_CONFIG["pytorch"]["LEARNING_RATE"], weight_decay = _CONFIG["pytorch"]["L2_WEIGHT_DECAY"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=_CONFIG["pytorch"]["LR_DECAY_FACTOR"], patience=_CONFIG["pytorch"]["LR_DECAY_PATIENCE"], verbose=True)

    NUM_EPOCHS = _CONFIG["pytorch"]["NUM_EPOCHS"]
    for epoch in range(NUM_EPOCHS):
        total_train_ADE = torch.zeros(1).to(device)
        total_train_FDE = torch.zeros(1).to(device)

        # training
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Training Epoch [{epoch +1}/{NUM_EPOCHS}]", colour='green')
        for _, batch in enumerate(batch_iterator):
            map_matrix, social_matrix, ground_truth = batch
            map_matrix, social_matrix, ground_truth = map_matrix.to(device), social_matrix.to(device), ground_truth.to(device)

            # only closest 100 segments
            map_matrix = map_matrix[:, :100]

            # jointly optimize for all ts by setting batch entries to different observation lengths
            social_matrix = mask_batch(social_matrix)

            optimizer.zero_grad()
            outputs = model(map_matrix, social_matrix) # N x 60 x 2

            pred_pos = compute_pos(outputs)
            train_ADE, train_FDE = calculate_ADE_FDE(pred_pos, ground_truth)
            loss = train_ADE # using ADE as loss function

            total_train_ADE += train_ADE
            total_train_FDE += train_FDE

            loss.backward()
            optimizer.step()

        average_train_ADE = total_train_ADE / len(batch_iterator)
        average_train_FDE = total_train_FDE / len(batch_iterator)
        _LOGGER_TRAIN.info(f"Tarining Epoch [{epoch+1}/{NUM_EPOCHS}], ADE: {average_train_ADE.item():.4f}, FDE: {average_train_FDE.item():.4f}")
        
        # validation
        torch.cuda.empty_cache()
        model.eval()
        batch_iterator = tqdm(val_loader, desc=f"Validation Epoch [{epoch +1}/{NUM_EPOCHS}]", colour='red')
        with torch.no_grad():
            total_val_ADE = torch.zeros(1).to(device)
            total_val_FDE = torch.zeros(1).to(device)
            for val_datas in batch_iterator:
                map_matrix, social_matrix, ground_truth = val_datas
                map_matrix, social_matrix, ground_truth = map_matrix.to(device), social_matrix.to(device), ground_truth.to(device)
        
                # only closest 100 segments
                map_matrix = map_matrix[:, :100]

                # jointly optimize for all ts by setting batch entries to different observation lengths
                social_matrix = mask_batch(social_matrix)

                outputs = model(map_matrix, social_matrix) # N x 60 x 2
                pred_pos = compute_pos(outputs)
                val_ADE, val_FDE = calculate_ADE_FDE(pred_pos, ground_truth)
                
                total_val_ADE += val_ADE
                total_val_FDE += val_FDE
        
        average_val_ADE = total_val_ADE / len(batch_iterator)
        average_val_FDE = total_val_FDE / len(batch_iterator)
        average_val_loss = average_val_ADE
        scheduler.step(average_val_loss)
        _LOGGER_TRAIN.info(f"Validation Epoch [{epoch+1}/{NUM_EPOCHS}], ADE: {average_val_ADE.item():.4f}, FDE: {average_val_FDE.item():.4f}")
        
        with open(_TRAINING_PATH / f"ADE_FDE_snapshot.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, average_train_ADE.item(), average_train_FDE.item(), average_val_ADE.item(), average_val_FDE.item()])
        
        if (epoch+1) % 2 == 0:
            weights_filename = f"model_epoch_{epoch+1}_{date}_.pth"
            torch.save(model.state_dict(), _TRAINING_PATH / weights_filename)

    final_weights_path = _TRAINING_PATH / f"model_epoch_{epoch+1}_{date}_.pth"
    torch.save(model.state_dict(), final_weights_path)

    with open(_TRAINING_PATH / "used_config.yaml", 'w') as file:
        yaml.dump(_CONFIG, file)

    _LOGGER_TRAIN.info("Training finished.")
    _LOGGER_TRAIN.info("Total Training time in h: %s \n", (time.time() - _START_TIME)/3600)

    return final_weights_path


if __name__ == "__main__":

    _START_TIME = time.time()

    # Configure logging
    date = datetime.now().strftime("%d%m%Y-%H%M%S")
    _TRAINING_PATH = Path(_CONFIG["path"]["TRAINING"]) / "snapshot" / date
    _TRAINING_PATH.mkdir(parents=True, exist_ok=True)
    log_filepath = _TRAINING_PATH / "Log_Train_Program.txt"

    _LOGGER_TRAIN = logging.getLogger('Logger_Train')
    _LOGGER_TRAIN.setLevel(logging.DEBUG)  # Definiere das Logging-Level

    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_formatter = logging.Formatter('%(asctime)s @Train_Program %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('@Train_Program %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    _LOGGER_TRAIN.addHandler(file_handler)
    _LOGGER_TRAIN.addHandler(console_handler)

    _LOGGER_TRAIN.info(_CONFIG["program"]["NAME"])
    _LOGGER_TRAIN.info("Version: %s", _CONFIG["program"]["VERSION"])
    _LOGGER_TRAIN.info(_CONFIG["program"]["DESCRIPTION"])
    _LOGGER_TRAIN.info("Starting Train Program...\n")
    
    final_weights_path = train_model()
    test_model(final_weights_path, _LOGGER_TRAIN)

    _LOGGER_TRAIN.info("Train Program finished.")




