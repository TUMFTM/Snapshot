# imports
import os
import sys
import re
import time
import logging
import yaml
import torch
from pathlib import Path
import numpy as np
from pqdm.processes import pqdm
from datetime import datetime
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

# local imports
sys.path.append(".")
from utils.map_features import vectorize_map
from utils.social_features import create_social_matrix_and_ground_truth

# global variables
with open(Path(__file__).parent.resolve() / "../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

_LOGGING_LEVEL = _CONFIG["general"]["LOGGING_LEVEL"]
_PATH_PREPROCESSED_DIR = Path(_CONFIG["path"]["PREPROCESSED_DATA"]) / _CONFIG["path"]["PREPROCESSED_DIR"]
_PATH_STORAGE_DIR = Path(_CONFIG["path"]["BENCHMARK_DATA"])
_START_TIME_PRE = time.time()

# Configure logging
date = datetime.now().strftime("%d%m%Y-%H%M%S")
log_dir = Path(_CONFIG["path"]["LOG"]) / "LOG"
log_dir.mkdir(parents=True, exist_ok=True)
log_filepath = log_dir / (f"Log_Data_Preprocessing_{date}_" + _CONFIG["program"]["NAME"] + "_V:"  +  _CONFIG["program"]["VERSION"] + ".txt")

_LOGGER_PREPROCESSING = logging.getLogger('Logger_Preprocessing')
_LOGGER_PREPROCESSING.setLevel(logging.getLevelName(_LOGGING_LEVEL))  # Definiere das Logging-Level

file_handler = logging.FileHandler(log_filepath, mode='w')
file_formatter = logging.Formatter('%(asctime)s @Data-Preprocessing %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('@Data-Preprocessing %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

_LOGGER_PREPROCESSING.addHandler(file_handler)
_LOGGER_PREPROCESSING.addHandler(console_handler)


def process_one_scenario(scenario_dir: Path) -> None:
    """
    Process a single scenario by performing data preprocessing steps.
    1. Load the scenario and map.
    2. Vectorize the map.
    3. Create the social matrix and ground truth.
    4. Save the preprocessed data

    Args:
        scenario_dir (Path): The directory path of the scenario.

    Global Variables:
        _path_preprocessed_dir (Path): The directory path for the preprocessed data.

    Returns:
        None

    Raises:
        If the map file cannot be loaded.
        If the scenario file cannot be loaded.
        If the map matrix has more than 560 vectors.
    """
    global _path_preprocessed_dir
    map_path = scenario_dir / f"log_map_archive_{scenario_dir.name}.json"
    try: 
        current_map = ArgoverseStaticMap.from_json(map_path)
        scenario_list = sorted((scenario_dir).glob("*.parquet"))
        for scenario_path in scenario_list:
            try:
                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                scored_track_id_list = re.findall(r'\d+', scenario.focal_track_id)
                for scored_track_id in scored_track_id_list:
                    current_dir = (_path_preprocessed_dir / f"{scenario.scenario_id}_{scored_track_id}")
                    current_dir.mkdir(parents=False, exist_ok=False)

                    map_matrix = vectorize_map(current_map, scenario, scored_track_id, _LOGGER_PREPROCESSING)
                    
                    social_matrix, ground_truth = create_social_matrix_and_ground_truth(scenario, scored_track_id)
                    if social_matrix.shape[0] < 8:
                        social_matrix = np.pad(social_matrix, ((0, 8-social_matrix.shape[0]), (0,0)), 'constant', constant_values=(0))

                    map_matrix = torch.from_numpy(map_matrix).to(torch.float32) # (560, 6)
                    social_matrix = torch.from_numpy(social_matrix).to(torch.float32) # (8, 10)
                    ground_truth = torch.from_numpy(ground_truth).to(torch.float32) # (60, 2)

                    torch.save(map_matrix, current_dir / "map_matrix.pt")
                    torch.save(social_matrix, current_dir / "social_matrix.pt")
                    torch.save(ground_truth, current_dir / "ground_truth.pt")
            except Exception as e:
                _LOGGER_PREPROCESSING.error(f"Error processing scenario {scenario_path}: {e}")
    except Exception as e:
        _LOGGER_PREPROCESSING.error(f"Error loading map {map_path}: {e}")


# Main Class
class Data_Preprocessing():

    def __init__(self):

        _LOGGER_PREPROCESSING.info(_CONFIG["program"]["NAME"])
        _LOGGER_PREPROCESSING.info("Version: %s", _CONFIG["program"]["VERSION"])
        _LOGGER_PREPROCESSING.info(_CONFIG["program"]["DESCRIPTION"])
    
        _LOGGER_PREPROCESSING.info("Starting Data Preprocessing... \n")

        self.path_storage_dir = Path(_CONFIG["path"]["PREPROCESSED_DATA"])
        _LOGGER_PREPROCESSING.info(f'Loading Data from Directory: {_CONFIG["path"]["BENCHMARK_DATA"]}/pedestrian_benchmark/')
        _LOGGER_PREPROCESSING.info(f"Save Data in Directory: {self.path_storage_dir}/preprocessed \n")

        self.folder_list = _CONFIG["general"]["WHICH_DATA_FOLDER"]
        if _CONFIG["general"]["CREATE_NEW_TEST_SET"] or _CONFIG["general"]["CREATE_TEST_SET_USING_ID_LIST"]:
            self.folder_list.append("test")

        for folder in self.folder_list:
            _LOGGER_PREPROCESSING.info("Preprocessing scenarios in %s...", folder)
            self.preprocess_data(folder)

        _LOGGER_PREPROCESSING.info("Data Preprocessing Finished")
        _LOGGER_PREPROCESSING.info("Total time in min: %s \n \n \n", (time.time() - _START_TIME_PRE) / 60)


    def preprocess_data(self, folder:str):
        """
        Preprocess the data by performing the following steps:
        1. Create the preprocessed directory.
        2. Load the scenarios.
        3. Process each scenario in parallel.


        Args:
            folder (str): The folder containing the scenarios.

        Global Variables:
            _PATH_PREPROCESSED_DIR (Path): The directory path for the preprocessed data.

        Returns:
            None

        Raises:
            RuntimeError: If no files are found in the scenario directory.
        """

        global _path_preprocessed_dir
        _path_preprocessed_dir = _PATH_PREPROCESSED_DIR / folder
        _path_preprocessed_dir.mkdir(parents=True, exist_ok=False)
        scenario_dir = _PATH_STORAGE_DIR / "pedestrian_benchmark" / folder
        _LOGGER_PREPROCESSING.info("Reading scenarios from: %s", scenario_dir)
        all_scenario_dir = sorted(scenario_dir.glob("*/"))
        _LOGGER_PREPROCESSING.info("Found %s scenarios in %s", len(all_scenario_dir), scenario_dir)

        if not all_scenario_dir:
            _LOGGER_PREPROCESSING.critical("No files found in: %s", scenario_dir)
            raise RuntimeError("A critical error has occurred. The program will terminate.")
        else: 
            num_cores = _CONFIG["general"]["NUM_CORES"]
            if num_cores == 1:
                _LOGGER_PREPROCESSING.info("DEBUG MODE: Only processing on one core!")
            else: 
                if num_cores < 1:
                    num_cores = os.cpu_count() + num_cores
                _LOGGER_PREPROCESSING.info(f"Parallel processing on {num_cores} cores!")
            pqdm(all_scenario_dir, process_one_scenario, n_jobs=num_cores, desc="Processing Folder: " + folder )
            _LOGGER_PREPROCESSING.info("Finished processing scenarios in %s \n", folder)


if __name__ == "__main__":
    _ = Data_Preprocessing()

