# imports
import torch
import yaml
import sys
import os
import logging
import math
import re
import random
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

# local imports
sys.path.append(".")
from utils.plot_functions import *
from utils.map_features import *
from utils.social_features import *
from utils.evaluation import compute_pos
from model.snapshot.snapshot import Snapshot

# global variables
with open(Path(__file__).parent.resolve() / "../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

# Configure logging
_LOGGER_VISUALIZATION = logging.getLogger('Logger_Visualization')
_LOGGER_VISUALIZATION.setLevel(logging.DEBUG)  # Definiere das Logging-Level
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('@Visualization %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
_LOGGER_VISUALIZATION.addHandler(console_handler)


def plot_vec_tracks(data: list, color: str , ax, radius: int):
    for elem in data:
        ax.scatter(elem[0::2]/radius, elem[1::2]/radius, color=color, s=5)
        

class Visualization():
    def __init__(self) -> None:

        parser = argparse.ArgumentParser(description="Tool for visualizing the vectorized map.")

        # add arguments
        parser.add_argument('--radius', type=int, default=None, help='Radius for the vectorization of the map. Default is taken from the config file.')
        parser.add_argument('--path', type=str, default=None, help='Path to the scenario folder. If not given, a random scenario (test folder) will be chosen.')
        parser.add_argument('--model', type=str, default=None, help='Path to the model weights. If not given, the main Snapshot model is used.')
        # Parse die Argumente
        args = parser.parse_args()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        # load model
        self.model = Snapshot().to(self.device)
        if args.model:
            weights = torch.load(args.model, map_location=self.device)
        else:
            weights = torch.load("./model/weights/snapshot_main.pth", map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()

        if args.path:
            _LOGGER_VISUALIZATION.info("Using scenario %s", args.path)
        else:
            _LOGGER_VISUALIZATION.info("Using random scenario.")
        self.run(args.radius, args.path)
        _LOGGER_VISUALIZATION.info("Finished visualization.")


    def run(self, radius = None, path = None):
        if path:
            target_dir = Path(path)
        else:
            path_filtered_dir = Path(_CONFIG["path"]["BENCHMARK_DATA"]) / "pedestrian_benchmark" / "test"
            all_scenario_folders = sorted(path_filtered_dir.glob("*/"))
            i = random.randint(0, len(all_scenario_folders)-1)
            target_dir = all_scenario_folders[i]

        path_output_dir = Path(_CONFIG["path"]["VIS_OUTPUT"]) / "images" / target_dir.name
        path_output_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER_VISUALIZATION.info("Output directory: %s", path_output_dir)

        map_file = list(target_dir.glob("*.json"))
        all_scenario_files = sorted(target_dir.glob("*.parquet"))
        scenario = scenario_serialization.load_argoverse_scenario_parquet(all_scenario_files[0])
        current_map = ArgoverseStaticMap.from_json(map_file[0])

        scored_track_id_list = re.findall(r'\d+', scenario.focal_track_id)
        scored_track_id = scored_track_id_list[0]
        if radius:
            map_matrix = vectorize_map(current_map, scenario, scored_track_id, _LOGGER_VISUALIZATION, radius, sorting=False)
        else:
            map_matrix = vectorize_map(current_map, scenario, scored_track_id, _LOGGER_VISUALIZATION, sorting=False)
            radius = max(_CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"], _CONFIG["vectorization"]["RADIUS_PED_CROSS"], _CONFIG["vectorization"]["RADIUS_LANE_SEG"])
        social_matrix, ground_truth = create_social_matrix_and_ground_truth(scenario, scored_track_id)

        focal_ped = social_matrix[social_matrix[:,0] == -1][:,1:]
        other_cars = social_matrix[social_matrix[:,0] == 0.6][:,1:]
        other_bus = social_matrix[social_matrix[:,0] == 1][:,1:]
        other_peds = social_matrix[social_matrix[:,0] == -0.6][:,1:]
        other_motorbikes = social_matrix[social_matrix[:,0] == 0.2][:,1:]
        other_cyclists = social_matrix[social_matrix[:,0] == -0.2][:,1:]

        # fig, ax = plt.subplots()
        # plot_map(ax, current_map)
        # plot_tracks(scenario, ax)
        # ax.set_aspect('auto')
        # plt.savefig(path_output_dir / "map.png",dpi=400)
        # plt.show()

        fig, ax = plt.subplots()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plot_vec_map(ax, map_matrix)

        # get prediction
        map_matrix = vectorize_map(current_map, scenario, scored_track_id, _LOGGER_VISUALIZATION, radius)[:, :100]
        map_matrix = torch.from_numpy(map_matrix[None, :, :].astype('float32')).to(self.device)
        social_matrix = torch.from_numpy(social_matrix[None, :, :].astype('float32')).to(self.device)
        outputs = self.model(map_matrix, social_matrix) # N x 60 x 2
        pred_pos = compute_pos(outputs)[0].detach().numpy()

        ax.scatter(ground_truth[:,0]/radius, ground_truth[:,1]/radius, color="b", s=1)
        ax.scatter(pred_pos[:,0]/radius, pred_pos[:,1]/radius, color="g", s=1)  
        ax.scatter(focal_ped[0][0::2]/radius, focal_ped[0][1::2]/radius, color="r", s=1)

        plot_vec_tracks(other_cars, "#FFC1C1", ax, radius)
        plot_vec_tracks(other_bus, "#EE9A49", ax, radius)
        plot_vec_tracks(other_motorbikes, "#D3E8EF", ax, radius)
        plot_vec_tracks(other_cyclists, "#D3E8EF", ax, radius)
        plot_vec_tracks(other_peds, "y", ax, radius)

        ax.set_aspect('equal')
        plt.savefig(path_output_dir / "scenario.png",dpi=400)
        #plt.show()




if __name__ == "__main__":
    vis = Visualization()
