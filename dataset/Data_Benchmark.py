# imports
import os
import copy
import logging
import random
import yaml
import time
import shutil
from pathlib import Path
import pandas as pd
from pqdm.processes import pqdm
from typing import List, Tuple
from datetime import datetime
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (ArgoverseScenario, ObjectType, TrackCategory, Track)

# global variables
with open("config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

_LOGGING_LEVEL = _CONFIG["general"]["LOGGING_LEVEL"]
_OBSERVATION_LENGTH = _CONFIG["samples"]["OBSERVATION_LENGTH"]
_START_TIME_FIL = time.time()

# Configure logging
date = datetime.now().strftime("%d%m%Y-%H%M%S")
log_dir = Path(_CONFIG["path"]["LOG"]) / "LOG"
log_dir.mkdir(parents=True, exist_ok=True)
log_filepath = log_dir / (f"Log_Data_Benchmark_{date}_" + _CONFIG["program"]["NAME"] + "_V:"  +  _CONFIG["program"]["VERSION"] + ".txt")

_LOGGER_BENCHMARK = logging.getLogger('Logger_Benchmark')
_LOGGER_BENCHMARK.setLevel(logging.getLevelName(_LOGGING_LEVEL))  # Definiere das Logging-Level

file_handler = logging.FileHandler(log_filepath, mode='w')
file_formatter = logging.Formatter('%(asctime)s @Data-Benchmark %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('@Data-Benchmark %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

_LOGGER_BENCHMARK.addHandler(file_handler)
_LOGGER_BENCHMARK.addHandler(console_handler)


# Helpfunctions

def filter_one_scenarios(scenario_path: Path) -> None:
    """
    Filters out non-pedestrian scenarios from the given folder by performing the following steps:
    1. Load the scenario.
    2. Check if the scenario contains pedestrian tracks.
    3. Remove the directory if it does not contain any pedestrian tracks.

    Args:
        scenario_path (Path): The path to the scenario file.

    Global Variables:
        None

    Returns:
        None

    Raises:
        If an error occurs while processing the file, the error message is logged.
    """

    try:
        scenario = pd.read_parquet(scenario_path)
        if 'pedestrian' not in scenario.object_type.values:
            # Remove the directory 
            shutil.rmtree(scenario_path.parent)
    except Exception as e:
        _LOGGER_BENCHMARK.error("Error processing file %s: %s", scenario_path, e)   




def process_one_scenario_timestep(scenario: ArgoverseScenario, _start_timestep: int, _end_timestep: int) -> ArgoverseScenario:
    """
    Processes a single scenario based on the given start and end timesteps by performing the following steps:
    1. Copy the scenario.
    2. Set the scenario ID.
    3. Set the timestamps.
    4. Process each track in the scenario.
    5. Return the processed scenario.

    Args:
        scenario (ArgoverseScenario): The input scenario.
        start_timestep (int): The starting timestep for the sample.
        end_timestep (int): The ending timestep for the sample.

    Global Variables:
        _OBSERVATION_LENGTH (int): The length of the observation period.

    Returns:
        ArgoverseScenario: The processed scenario.

    Raises:
        None
    """

    def process_one_track(track: Track) -> Track:
        """
        Processes a single track based on the given start and end timesteps by performing the following steps:
        1. Remove tracks that are not at all observable in the observation period.
        2. Process tracks that are completely observable in the observation period.
        3. Process tracks that are completely observable in the (observation + prediction) period and are pedestrians.
        4. Process tracks that are completely observable in the (observation) period but not in the (observation + prediction) period or are not pedestrians.
        5. Process tracks that begin before the observation period and end in the observation period.
        6. Process tracks that begin in the observation period and end during the observation period.
        7. Process tracks that begin before the observation period and end after the observation period.

        Args:
            track: The track object to be processed.

        Global Variables:
            _OBSERVATION_LENGTH (int): The length of the observation period.
            _start_timestep (int): The start of the observation period.
            _obs_threshold (int): The end of the observation period.
            _end_timestep (int): The end of the prediction period.
            _id (int): The track ID.
            _scored_track_id_str (str): The scored track ID string.

        Returns:
            The processed track object.

        Raises:
            None
    
        Logic:

                            observation      prediction
                start_timestep   obs_threshold       end_timestep
                        --obs_length- 
                        -----------sample_length-----------
                        0 1     ... 9 10 11   ...        69
                        |---- 1s ----|-------- 6s ---------| 
        1. remove     --|                |--
        2. if           <---------------->
        3. if 2 and     <-------------------------------------->          and pedestrian
        4. if 2 and     <-----------------------|                         or not pedestrian
        5. if           <----------|
        6. if                |-----|
        7. if                |------------->
        """

        # 1. remove tracks that are not at all observable in the observation period
        if track.object_states[-1].timestep < _start_timestep or track.object_states[0].timestep > _obs_threshold:
            return None
        else:
            if track.track_id != "AV":
                global _id
                track.track_id = str(_id).zfill(3)
                _id +=1
        # 2. tracks that are completly observable in the observation period
        if track.object_states[0].timestep <= _start_timestep and track.object_states[-1].timestep >= _obs_threshold:

            # 3. tracks that are completly observable in the (observation + prediction) period and are pedestrians
            if track.object_type == ObjectType.PEDESTRIAN and track.object_states[-1].timestep >= _end_timestep:
                track.category = TrackCategory.SCORED_TRACK
                global _scored_track_id_str
                _scored_track_id_str += f"{track.track_id}, "
                # remove states that are not in the (observation + prediction) period
                track.object_states = track.object_states[(_start_timestep - track.object_states[0].timestep):(_end_timestep - track.object_states[0].timestep) + 1]
                # set observed to False for states that are not in the observation period
                [setattr(track.object_states[i], 'observed', False) for i in range(_OBSERVATION_LENGTH, len(track.object_states))]

            # 4. tracks that are completly observable in the (observation) period but not in the (observation + prediction) period or are not pedestrians
            else:
                if track.track_id != "AV":
                    track.category = TrackCategory.UNSCORED_TRACK
                else:
                    track.category = TrackCategory.FOCAL_TRACK
                # remove states that are not in the observation period
                track.object_states = track.object_states[(_start_timestep - track.object_states[0].timestep):(_obs_threshold - track.object_states[0].timestep) + 1]

        # 5. tracks that beginn before the observation period and end in the observation period
        elif track.object_states[0].timestep <= _start_timestep and track.object_states[-1].timestep < _obs_threshold:
            track.category = TrackCategory.TRACK_FRAGMENT
            # remove states that are before the observation period
            track.object_states = track.object_states[(_start_timestep - track.object_states[0].timestep):]

        # 6. tracks that beginn in the observation period and end during the observation period
        elif track.object_states[0].timestep > _start_timestep and track.object_states[-1].timestep < _obs_threshold:
            track.category = TrackCategory.TRACK_FRAGMENT

        # 7. tracks that beginn before the observation period and end after the observation period
        elif track.object_states[0].timestep > _start_timestep and track.object_states[-1].timestep >= _obs_threshold:
            track.category = TrackCategory.UNSCORED_TRACK
            # remove states that are after the observation period
            track.object_states = track.object_states[:(_obs_threshold - track.object_states[0].timestep) + 1]

        return track

    
    scenario_sample = copy.deepcopy(scenario)
    scenario_sample.scenario_id = f"{scenario.scenario_id}_{_start_timestep}"
    scenario_sample.timestamps_ns = scenario.timestamps_ns[_start_timestep:_end_timestep+1] 
    global _id
    _id = 0
    global _scored_track_id_str
    _scored_track_id_str = ""
    _obs_threshold = _start_timestep + _OBSERVATION_LENGTH - 1 # end of observation period of 1s
    scenario_sample.tracks = list(filter(None, map(process_one_track, scenario_sample.tracks)))
    scenario_sample.focal_track_id = _scored_track_id_str
    return scenario_sample
    
    

def process_one_sample(scenario_path:Path) -> None:
    """
    Processes a single scenario by performing the following steps:
    1. Load the scenario.
    2. Filter the tracks based on the given conditions.
    3. Map the filtered tracks to get the desired intervals.
    4. Create a storage directory for the scenario.
    5. Copy the scenario map to the storage directory.
    6. Calculate the sliding range of the window.
    7. Process the scenario based on the sliding range.
    
    Args:
        scenario_path (Path): The path to the scenario file.

    Global Variables:
        _path_storage_dir (Path): The path to the storage directory.

    Returns:
        None

    Raises:
        If an error occurs while processing the file, the error message is logged.
    """
    try:
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        track_intervals: List[Tuple[int, int]] = []
        
        # Filtering tracks that meet the conditions
        filtered_tracks = filter(
            lambda track: track.object_type == ObjectType.PEDESTRIAN and len(track.object_states) >= _CONFIG["samples"]["SAMPLE_LENGTH"],
            scenario.tracks
        )
        # Mapping the filtered tracks to get the desired intervals
        track_intervals = list(map(
            lambda track: [track.object_states[0].timestep, track.object_states[-1].timestep + 1],
            filtered_tracks
        ))

        if track_intervals:
            # Create a storage directory for the scenario
            global _path_storage_dir
            storage_dir = _path_storage_dir / scenario_path.parent.name
            storage_dir.mkdir(parents=False, exist_ok=False)

            # Copy the scenario map to the storage directory
            source_file = scenario_path.parent / f"log_map_archive_{scenario_path.parent.name}.json"
            destination_file = storage_dir / f"log_map_archive_{scenario_path.parent.name}.json"
            shutil.copy(source_file, destination_file)

            # calculate the sliding range of the window
            min_start_timestep = min(start for start, _ in track_intervals)
            max_end_timestep = max(end for _, end in track_intervals)
            window_sliding_range = (min_start_timestep, max_end_timestep - _CONFIG["samples"]["SAMPLE_LENGTH"])
            
            # determine whether the window covers any complete pedestrian trajectory
            for start_timestep in range(window_sliding_range[0], window_sliding_range[1] + 1, _CONFIG["samples"]["DURATION_INTERVAL"]):
                end_timestep = start_timestep + _CONFIG["samples"]["SAMPLE_LENGTH"] - 1
                window_covers_track = any([start <= start_timestep and end >= end_timestep for start, end in track_intervals])

                if window_covers_track:
                    scenario_sample = process_one_scenario_timestep(scenario, start_timestep, end_timestep)
                    scenario_sample_path = storage_dir / f"scenario_{scenario_sample.scenario_id}.parquet"
                    scenario_serialization.serialize_argoverse_scenario_parquet(scenario_sample_path, scenario_sample)

    except Exception as e:
        _LOGGER_BENCHMARK.error(f"Error generating samples for scenario {scenario_path}: {e}")
        


# Main Class

class Data_Benchmark():
    def __init__(self) -> None:
        """
        Initializes the Data_Benchmark class.

        This method loads the configuration from a YAML file, sets up logging, and initializes the paths for data and storage directories.
        It also extracts pedestrian samples, filters pedestrian scenarios, generates samples, and generates train, val, and test splits.

        Parameters:
            None

        Returns:
            None
        """
        _LOGGER_BENCHMARK.info(_CONFIG["program"]["NAME"])
        _LOGGER_BENCHMARK.info("Version: %s", _CONFIG["program"]["VERSION"])
        _LOGGER_BENCHMARK.info(_CONFIG["program"]["DESCRIPTION"])

        _LOGGER_BENCHMARK.info("Starting Data Filtering... \n")
        # Initialize paths
        self.path_data_dir = Path(_CONFIG["path"]["DATA"])
        self.path_storage_dir = Path(_CONFIG["path"]["BENCHMARK_DATA"])

        _LOGGER_BENCHMARK.info("Data Directory: %s/", self.path_data_dir)
        _LOGGER_BENCHMARK.info("Storage Directory: %s/pedestrian_benchmark/", self.path_storage_dir)
        _LOGGER_BENCHMARK.info("Use Folder: %s \n", _CONFIG["general"]["WHICH_DATA_FOLDER"])


        # STEP 1: extract pedestrian samples
        _LOGGER_BENCHMARK.info("Extracting samples...")
        for folder in _CONFIG["general"]["WHICH_DATA_FOLDER"]:

            # remove scenarios without pedestrian tracks
            _LOGGER_BENCHMARK.info("Filtering pedestrian scenarios in %s...", folder)
            self.filter_pedestrian_scenarios(folder)

            # perform sliding window approach for each scenario to generate samples
            _LOGGER_BENCHMARK.info("Generating samples for %s...", folder)
            self.generate_samples(folder)

        # STEP 2: generate train, val and test splits
        if _CONFIG["general"]["CREATE_NEW_TEST_SET"]:
            _LOGGER_BENCHMARK.info("Generating NEW Test Set...")
            self.generate_train_val_test_splits(_CONFIG["general"]["SIZE_TEST_SET_IN_PERCENT"], 
                                                _CONFIG["general"]["SEED_TEST_SET"], 
                                                _CONFIG["general"]["WHICH_DATA_FOLDER"])
            
        if _CONFIG["general"]["CREATE_TEST_SET_USING_ID_LIST"]:
            _LOGGER_BENCHMARK.info("Generating Test Set using ID List at ./IDs_test_split.txt")
            self.generate_train_val_test_splits_with_textfile()

        _LOGGER_BENCHMARK.info("Data_Filtering finished!")
        _LOGGER_BENCHMARK.info("Total time in min: %s \n \n \n", (time.time() - _START_TIME_FIL)/60)




    def filter_pedestrian_scenarios(self, folder: str) -> None:
        """
        Filters out non-pedestrian scenarios from the given folder by performing the following steps:
        1. Get all scenario files in the folder.
        2. Process each scenario file.

        Args:
            folder (str): The name of the folder containing the scenarios.

        Global Variables:
            _CONFIG (dict): The configuration dictionary.

        Returns:
            None

        Raises:
            If no scenario files are found in the given folder, a critical error is raised.
        """
        
        # get all files in the folder
        _LOGGER_BENCHMARK.info("Reading files in %s...", folder)
        all_scenario_files = list((self.path_data_dir / folder).rglob("*.parquet"))
        _LOGGER_BENCHMARK.info("Found %d scenario files in %s", len(all_scenario_files), folder)
        if not all_scenario_files:
            _LOGGER_BENCHMARK.critical("No scenario files found in: %s during filtering", folder)
            raise RuntimeError("A critical error has occurred. The program will terminate.")
        else:
            
            num_cores = _CONFIG["general"]["NUM_CORES"]
            if num_cores == 1:
                _LOGGER_BENCHMARK.info("DEBUG MODE: Only processing on one core!")
            else: 
                if num_cores < 1:
                    num_cores = os.cpu_count() + num_cores
                _LOGGER_BENCHMARK.info(f"Parallel processing on {num_cores} cores!")
            pqdm(all_scenario_files, filter_one_scenarios, n_jobs=num_cores, desc="Processing Folder: " + folder) 
            
            # Get the remaining .parquet files in the folder
            rest_scenario_files = list((self.path_data_dir / folder).rglob("*.parquet"))
            _LOGGER_BENCHMARK.info("Deleted %d scenarios \n", len(all_scenario_files) - len(rest_scenario_files))



    def generate_samples(self, folder: str) -> None:
        """
        Generates samples for the given folder by performing the following steps:
        1. Get all scenario files in the folder.
        2. Process each scenario file.

        Args:
            folder (str): The folder name.

        Global Variables:
            None

        Returns:
            None

        Raises:
            If no scenario files are found in the given folder, a warning is logged.
        """

        # create storage directory    
        global _path_storage_dir
        _path_storage_dir = self.path_storage_dir / "pedestrian_benchmark" / folder
        _path_storage_dir.mkdir(parents=True, exist_ok=False)

        _LOGGER_BENCHMARK.info("Reading files in %s...", folder)
        all_scenario_files = list((self.path_data_dir / folder).rglob("*.parquet"))
        _LOGGER_BENCHMARK.info("Found %d scenario files in %s", len(all_scenario_files), folder)

        if not all_scenario_files:
            _LOGGER_BENCHMARK.warning("No scenario files found in: %s during sample generation", folder)

        else:
            num_cores = _CONFIG["general"]["NUM_CORES"]
            if num_cores == 1:
                _LOGGER_BENCHMARK.info("DEBUG MODE: Only processing on one core!")
            else: 
                if num_cores < 1:
                    num_cores = os.cpu_count() + num_cores
                _LOGGER_BENCHMARK.info(f"Parallel processing on {num_cores} cores!")
            pqdm(all_scenario_files, process_one_sample, n_jobs=num_cores, desc="Processing Folder: " + folder ) 
            _LOGGER_BENCHMARK.info("Generated samples for %s \n", folder)


    def generate_train_val_test_splits(self, size_test_set_in_percent: int, seed_test_set: int, which_data_folder: List[str]) -> None:
        """
        Generates train, validation, and test splits for the given dataset by performing the following steps:
        1. Create the test directory.
        2. Get all scenario files in the given folders.
        3. Randomly (seed given) select a percentage of the data for the test set.
        4. Move the selected data to the test directory.


        Args:
            path_storage_dir (Path): The path to the storage directory.
            size_test_set_in_percent (int): The percentage of data to be used for the test set.
            seed_test_set (int): The seed value for generating deterministic results.
            which_data_folder (List[str]): The list of folders containing the dataset.

        Global Variables:
            None

        Returns:
            None

        Raises:
            If no scenario files are found in the given folder, a warning is logged.
        """
        
        path_test_dir = self.path_storage_dir / "pedestrian_benchmark" / "test"
        path_test_dir.mkdir(parents=False, exist_ok=False)

        # define seed for deterministic results
        random.seed(seed_test_set) 

        for folder in which_data_folder:
            _LOGGER_BENCHMARK.info("Reading files in %s...", folder)
            all_scenario_files = sorted((self.path_storage_dir / "pedestrian_benchmark" / folder).glob("*/"))
            _LOGGER_BENCHMARK.info("Found %d scenario files in %s", len(all_scenario_files), folder)
            if not all_scenario_files:
                _LOGGER_BENCHMARK.warning("No scenario files found in: %s during test set creation", folder)
            else:
                num_test_samples = int(len(all_scenario_files) * size_test_set_in_percent / 100)
                _LOGGER_BENCHMARK.info("Creating test set with %d samples from %s", num_test_samples, folder)
                test_samples = random.sample(all_scenario_files, num_test_samples)
                list(map(lambda test_sample: shutil.move(test_sample, path_test_dir), test_samples))
                _LOGGER_BENCHMARK.info("Moved %d samples to test set", len(test_samples))




    def generate_train_val_test_splits_with_textfile(self) -> None:
        """
        Generate train, validation, and test splits based on a text file containing target IDs.

        Args:
            path_to_test_target_ids (str): The path to the text file containing target IDs.

        Global Variables:
            path_storage_dir (Path): The path to the storage directory.

        Returns:
            None

        Raises:
            None
        """
        path_test_dir = self.path_storage_dir / "pedestrian_benchmark" / "test"
        path_test_dir.mkdir(parents=False, exist_ok=False)

        # version with scenario IDs
        with open("./dataset/IDs_test_split.txt", "r") as file:
            unique_lines = set(map(str.strip, file.readlines()))


        for folder in ["train", "val"]:
            current_path = self.path_storage_dir / "pedestrian_benchmark" / folder
            all_scenario_files = [element.name for element in sorted(current_path.glob("*/"))]
            matching_elemts = unique_lines.intersection(all_scenario_files)
            _LOGGER_BENCHMARK.info("Found %d scenario files in %s", len(matching_elemts), folder)
            list(map(lambda matching_elemts: shutil.move(str(current_path / matching_elemts), path_test_dir), matching_elemts))
            _LOGGER_BENCHMARK.info("Moved %d samples to test set", len(matching_elemts))

        _LOGGER_BENCHMARK.info("Finished generating test set \n")
        

if __name__ == "__main__":
    _ = Data_Benchmark()

