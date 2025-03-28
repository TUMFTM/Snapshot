# imports
import torch
import yaml
import sys
import logging
import re
import random
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

# local imports
sys.path.append(".")
from utils.map_features import vectorize_map
from utils.social_features import *
from utils.evaluation import compute_pos
from utils.plot_functions import *

from model.snapshot.snapshot import Snapshot

# Color Palette
blue = "#0065BD"
green = "#2ECC71" 

# Configure logging
_LOGGER_VISUALIZATION = logging.getLogger('Logger_Visualization')
_LOGGER_VISUALIZATION.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('@Visualization %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
_LOGGER_VISUALIZATION.addHandler(console_handler)

# global variables
with open(Path(__file__).parent.resolve() / "../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file) 


class Visualization():
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Tool for visualizing the vectorized map.")
        parser.add_argument('--radius', type=int, default=None,
                            help='Radius for the vectorization of the map. Default is taken from the config file.')
        parser.add_argument('--path', type=str, default=None,
                            help='Path to the scenario folder. If not given, a random scenario (test folder) will be chosen.')
        parser.add_argument('--model', type=str, default=None,
                            help='Path to the model weights. If not given, the main Snapshot model is used.')
        args = parser.parse_args()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        # Load model
        self.model = Snapshot().to(self.device)
        if args.model:
            weights = torch.load(args.model, map_location=self.device)
        else:
            weights = torch.load("./model/weights/snap_10obs_robust_2.pth", map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()

        if args.path:
            _LOGGER_VISUALIZATION.info("Using scenario %s", args.path)
        else:
            _LOGGER_VISUALIZATION.info("Using random scenario.")
        self.run(args.radius, args.path)
        _LOGGER_VISUALIZATION.info("Finished visualization.")

    def run(self, radius=None, path=None):
        if path:
            target_dir = Path(path)
        else:
            path_filtered_dir = Path(_CONFIG["path"]["BENCHMARK_DATA"]) / "pedestrian_benchmark" / "test"
            all_scenario_folders = sorted(path_filtered_dir.glob("*/"))
            i = random.randint(0, len(all_scenario_folders) - 1)
            target_dir = all_scenario_folders[i]

        map_file = list(target_dir.glob("*.json"))
        all_scenario_files = sorted(target_dir.glob("*.parquet"))
        scenario = scenario_serialization.load_argoverse_scenario_parquet(all_scenario_files[0])
        current_map = ArgoverseStaticMap.from_json(map_file[0])

        scored_track_id_list = re.findall(r'\d+', scenario.focal_track_id)
        scored_track_id = scored_track_id_list[0]
        if radius:
            map_matrix = vectorize_map(current_map, scenario, scored_track_id,
                                       _LOGGER_VISUALIZATION, radius, apl_sorting=False, max_num_vectors=2000)
        else:
            map_matrix = vectorize_map(current_map, scenario, scored_track_id,
                                       _LOGGER_VISUALIZATION, apl_sorting=False, max_num_vectors=2000)
            radius = max(_CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"],
                         _CONFIG["vectorization"]["RADIUS_PED_CROSS"],
                         _CONFIG["vectorization"]["RADIUS_LANE_SEG"])
        social_matrix, ground_truth = create_social_matrix_and_ground_truth(scenario, scored_track_id)

        # create the figure and set up axes
        fig, ax = plt.subplots()
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        plot_vec_map(ax, map_matrix)

        # get prediction
        map_matrix_for_pred = vectorize_map(current_map, scenario, scored_track_id,
                                            _LOGGER_VISUALIZATION, radius)[:100]
        map_matrix_for_pred = torch.from_numpy(map_matrix_for_pred[None, :, :].astype('float32')).to(self.device)
        social_matrix_t = torch.from_numpy(social_matrix[None, :, :].astype('float32')).to(self.device)
        outputs = self.model(map_matrix_for_pred, social_matrix_t)  # N x 60 x 2
        pred_pos = compute_pos(outputs)[0].detach().numpy()

        # plot trajectories
        # Ground truth future of the focal pedestrian (blue dashed line)
        ax.plot(ground_truth[:, 0] / radius, ground_truth[:, 1] / radius,
                color=blue, linestyle="--", linewidth=1, label="Ground truth")
        # Prediction (green dotted line)
        ax.plot(pred_pos[:, 0] / radius, pred_pos[:, 1] / radius,
                color=green, linestyle="dotted", linewidth=1.2, label="Prediction", zorder=6)

        # plot agents
        for agent in social_matrix:
            if agent[0] != 0:
                plot_agent(ax, agent)
            else:
                break

        ax.set_aspect('equal')
        ax.set_ylabel("Coordinates in m")
        ax.set_xlabel("Coordinates in m")
        ax.legend(loc="lower right")

        # transform to metric frame
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x*20:g}"))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, pos: f"{y*20:g}"))

        file = str(target_dir.name) + "_scenario.svg"
        plt.title("Scenario " + str(target_dir.name), fontsize=8)
        plt.savefig(Path(_CONFIG["path"]["VIS_OUTPUT"]) / "images" / file, dpi=400, bbox_inches='tight')
        # plt.show()

if __name__ == "__main__":
    vis = Visualization()
