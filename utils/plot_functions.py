# imports
import numpy as np
import math
from typing import Final, Tuple, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from av2.map.map_api import ArgoverseStaticMap
import av2.rendering.vector as vector_plotting_utils
from av2.utils.typing import NDArrayFloat
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory
)


# global variables
_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_ESTIMATED_BUS_LENGTH_M: Final[float] = 10.0
_ESTIMATED_BUS_WIDTH_M: Final[float] = 2.2
_BOUNDING_BOX_ZORDER: Final[int] = (
    100  # Ensure actor bounding boxes are plotted on top of all map elements
)

def plot_vec_map(ax, map_matrix):
    driv_area = map_matrix[map_matrix[:,0] == -0.9][:,1:]
    ped_crossing = map_matrix[map_matrix[:,0] == 0.3][:,1:]
    lanes = map_matrix[map_matrix[:,0] == -0.3][:,1:]

    if driv_area.size != 0:
        key = driv_area[0][0]
        current_matrix = []
        for row in driv_area:
            if row[0] == key:
                current_matrix.append(row)
            else:
                current_matrix = np.array(current_matrix)
                ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="b", linewidth=0.5)
                ax.fill(current_matrix[:,1], current_matrix[:,2], color='b', alpha=0.2)
                key = row[0]
                current_matrix = [row]
        current_matrix = np.array(current_matrix)
        ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="b", linewidth=0.5)
        ax.fill(current_matrix[:,1], current_matrix[:,2], color='b', alpha=0.2)

    if ped_crossing.size != 0:
        key = ped_crossing[0][0]
        current_matrix = []
        for row in ped_crossing:
            if row[0] == key:
                current_matrix.append(row)
            else:
                current_matrix = np.array(current_matrix)
                ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="m", linewidth=0.5)
                ax.fill(current_matrix[:,1], current_matrix[:,2], color='m', alpha=0.2)
                key = row[0]
                current_matrix = [row]
        current_matrix = np.array(current_matrix)
        ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="m", linewidth=0.5)
        ax.fill(current_matrix[:,1], current_matrix[:,2], color='m', alpha=0.2)

    if lanes.size != 0:
        key = lanes[0][0]
        current_matrix = []
        for row in lanes:
            if row[0] == key:
                current_matrix.append(row)
            else:
                current_matrix = np.array(current_matrix)
                ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="#E0E0E0", linewidth=0.5)
                key = row[0]
                current_matrix = [row]
        current_matrix = np.array(current_matrix)
        ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="#E0E0E0", linewidth=0.5)


def _plot_actor_bounding_box(
    ax: plt.Axes, cur_location: NDArrayFloat, heading: float,
    color: str, bbox_size: Tuple[float, float]) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box



    """
    ^
    |   --------
    |  |  Auto  |
    |   --------
     ---------------->
    Länge Diagonale  = d
    Winkel zwischen Diagonale und X-Achse = theta_2
    Heading entspricht Winkel zwischen X-Achse und Fahrzeuglängsachse in mathematisch positiver Richtung
      im Fall oben 0 Radiant
    """


    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        angle=np.degrees(heading),
        color=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)


def plot_tracks(szenario: ArgoverseScenario, ax: plt.Axes) -> None:
    for track in szenario.tracks:

        if track.track_id == 'AV':
            _plot_actor_bounding_box(
                ax,
                track.object_states[-1].position,
                track.object_states[-1].heading,
                "#FF0000",
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )

        elif track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                track.object_states[-1].position,
                track.object_states[-1].heading,
                "#FFC1C1",
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )

        elif track.object_type == ObjectType.BUS:
            _plot_actor_bounding_box(
                ax,
                track.object_states[-1].position,
                track.object_states[-1].heading,
                "#EE9A49",
                (_ESTIMATED_BUS_LENGTH_M, _ESTIMATED_BUS_WIDTH_M),
            )

        elif (
            track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST):
            _plot_actor_bounding_box(
                ax,
                track.object_states[-1].position,
                track.object_states[-1].heading,
                "#D3E8EF",
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )

        
        if track.category == TrackCategory.SCORED_TRACK:
            for i in range(10):
                plt.scatter(track.object_states[i].position[0], track.object_states[i].position[1], color = "#ECA25B", s=5)

            for i in range(10, len(track.object_states)):
                plt.scatter(track.object_states[i].position[0], track.object_states[i].position[1], color = "#007672", s=5)


def plot_map(ax, static_map: ArgoverseStaticMap) -> None:
    # Plot the drivable_area
    for _, da in static_map.vector_drivable_areas.items():
        vector_plotting_utils.draw_polygon_mpl(ax, da.xyz, color="b", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(da.xyz, ax, color="b", alpha=0.2)

    for _, pc in static_map.vector_pedestrian_crossings.items():
        vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="m", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(pc.polygon, ax, color="m", alpha=0.2)
        
    for _, ls in static_map.vector_lane_segments.items():
        vector_plotting_utils.draw_polygon_mpl(ax, ls.polygon_boundary, color="#E0E0E0", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(ls.polygon_boundary, ax, color="#E0E0E0", alpha=0.2)









