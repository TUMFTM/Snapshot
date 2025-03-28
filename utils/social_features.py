# imports
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario, Track, ObjectType, TrackCategory)
from av2.utils.typing import NDArrayFloat
from pathlib import Path
from typing import Tuple, List
import yaml
import numpy as np
import math

# local imports
from .extract_heading import obtain_current_heading_speed_from_data, obtain_exact_heading_speed_pos, obtain_exact_velocity

# global variables
_MOVER_TYPE_DICT = {
    'FOCAL_PEDESTRIAN': -1.0,
    ObjectType.PEDESTRIAN: -0.6,
    ObjectType.CYCLIST: -0.2,
    ObjectType.MOTORCYCLIST: 0.2,
    ObjectType.VEHICLE: 0.6,
    ObjectType.BUS: 1.0,
}

# global variables
with open(Path(__file__).parent.resolve() / "../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

_OBSERVATION_LENGTH = _CONFIG["samples"]["OBSERVATION_LENGTH"]
_DISTANCE_BASED_SELECTION = _CONFIG["samples"]["DISTANCE_BASED_SELECTION"]
_ROTATION = _CONFIG["samples"]["ROTATION"]
_MAX_RADIUS_AGENTS = _CONFIG["samples"]["MAX_RADIUS_AGENTS"]
_MAX_NUM_AGENTS = _CONFIG["samples"]["MAX_NUM_AGENTS"]

def _calculate_vtr_angle(vtr1: np.ndarray, vtr2: np.ndarray):
    cosTh = np.dot(vtr1, vtr2)
    sinTh = np.cross(vtr1, vtr2)
    del_alpha = np.arctan2(sinTh, cosTh)
    return del_alpha


def calculate_cognitive_metrics(pedestrian_position_t: NDArrayFloat, pedestrian_velocity_t: NDArrayFloat,
    mover_position_t: NDArrayFloat, mover_velocity_t: NDArrayFloat):

    vtr_p_rel = mover_position_t - pedestrian_position_t
    vtr_v_rel = mover_velocity_t - pedestrian_velocity_t

    delta_alpha = _calculate_vtr_angle(vtr_p_rel, (vtr_p_rel + vtr_v_rel))

    if (np.linalg.norm(vtr_v_rel) == 0.):
        ttca = 0.
    else:
        ttca = - np.dot(vtr_p_rel, vtr_v_rel) / (np.linalg.norm(vtr_v_rel)**2)

    # object away from pedestrian
    if ttca < 0.:
        ttca = 0.
        _dca = vtr_p_rel
    else:
        _dca = vtr_p_rel + ttca * vtr_v_rel
    dca = np.linalg.norm(_dca)

    return delta_alpha, ttca, dca, np.linalg.norm(vtr_v_rel), np.linalg.norm(vtr_p_rel)



def calculate_collision_risk(delta_alpha: float, ttca: float, dca: float, s_rel: float, d_rel: float) -> float:
    
    if s_rel < 0.5:
        ttca = 0.
        dca = d_rel
        
    rel_delta_alpha = delta_alpha / (2.0**2)
    rel_ttca = ttca / (18.0**2)
    rel_dca = dca / (10.0**2)
    
    collision_risk = np.exp(-0.5 * (delta_alpha * rel_delta_alpha + ttca * rel_ttca + dca * rel_dca))
    return collision_risk


def create_observation_and_ground_truth(focal_ped_track: Track, focal_ped_position: Tuple[float,float], rotation_matrix: NDArrayFloat, apl_rotation: bool) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Creates the observation and ground truth for a given focal pedestrian track by performing the following steps:
    1. Extract the positions of the object states in the track.
    2. Calculate the relative positions of the object states with respect to the focal pedestrian position.
    3. Rotate the filtered relative positions using the rotation matrix.
    3. Separate the relative positions into observation and ground truth.

    Args:
        focal_ped_track (Track): The track of the focal pedestrian.
        focal_ped_position (Tuple[float, float]): The position of the focal pedestrian.
        rotation_matrix (NDArrayFloat): The rotation matrix.

    Global Variables:
        _OBSERVATION_LENGTH (int): The length of the observation sequence.

    Returns:
        Tuple[NDArrayFloat, NDArrayFloat]: A tuple containing the observation and ground truth array
            with shape (N, 2) where N is the number of prediction states. 
            Each row contains the x and y coordinates.

    Raises:
        None
    """
    positions_xy =  np.array([object_state.position for object_state in focal_ped_track.object_states])

    relative_pos_xy = positions_xy - focal_ped_position
    if apl_rotation:
        relative_pos_xy = np.dot(relative_pos_xy, rotation_matrix)


    observation = relative_pos_xy[:_OBSERVATION_LENGTH]
    ground_truth = relative_pos_xy[_OBSERVATION_LENGTH:]
    
    return observation, ground_truth



def create_mover_matrix(scenario: ArgoverseScenario, focal_track_id: str, focal_ped_pos_end_obs: Tuple[float,float], theta: float, rotation_matrix: NDArrayFloat, apl_rotation: bool) -> NDArrayFloat:
    """
    Create a mover matrix based on the given scenario and parameters by performing the following steps:
    1. Extract all tracks that fit the criteria.
    2. Extract the relative positions of the last observation state of each track with 
        respect to the focal pedestrian and rotate them.
    3. Filter the relative positions to only include the tracks that are in front of the focal pedestrian.
    4. Extract the heading and speed between the last two object states of each track.
    5. Extract the heading and speed between the first and last object states of each track.
    6. Extract the object types of the tracks.
    7. Create the mover matrix and sort it based on the distance from the focal pedestrian.


    Args:
        scenario (scenario_serialization.ArgoverseScenario): The scenario containing the tracks.
        focal_track_id (str): The ID of the focal track.
        focal_ped_pos_end_obs (Tuple[float, float]): The position of the focal pedestrian's last observation.
        theta (float): The rotation angle in radians.
        rotation_matrix (NDArrayFloat): The rotation matrix.

    Global Variables:
        _MOVER_TYPE_DICT (dict): Dictionary mapping object types to their corresponding values.

    Returns:
        NDArrayFloat: The mover matrix. Shape (N, 21) where N is the number of tracks. Each row contains the following columns:
            - type (float): Type of the mover. -0.6 for pedestrians, -0.2 for cyclists, 0.2 for motorcyclists, 0.6 for vehicles, 1.0 for buses.
            - x (float): x-coordinate of the relative position of the last observation state.
            - y (float): y-coordinate of the relative position of the last observation state.
    Raises:
        None

    """
    # Extract all tracks that fit following criteria:
    # - Object type is VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS
    # - Category is not TRACK_FRAGMENT
    # - Track ID is not the focal track ID
    track_list = np.array([track for track in scenario.tracks if track.object_type in 
              {ObjectType.VEHICLE, ObjectType.PEDESTRIAN, ObjectType.MOTORCYCLIST, 
               ObjectType.CYCLIST, ObjectType.BUS} 
               and track.category != TrackCategory.TRACK_FRAGMENT 
               and track.track_id != focal_track_id])

    if len(track_list) > 0:
        # Extract the relative positions of the last observation state 
        # of each track with respect to the focal pedestrian and rotate them. Shape (N, 2)
        pos_end_obs_list = np.array([track.object_states[-1].position for track in track_list])
        rel_pos_end_obs_list = pos_end_obs_list - focal_ped_pos_end_obs


        # Filter the relative positions to only include the tracks that are near the focal pedestrian 
        filtered_indices = np.where(np.linalg.norm(rel_pos_end_obs_list, axis=1) < _MAX_RADIUS_AGENTS)[0]

        if len(filtered_indices) > 0:
            filtered_track_list = track_list[filtered_indices]

            rot_rel_pos_mover_tracks = np.zeros((len(filtered_track_list), 20)) # pad missing values with zeros
            for i, track in np.ndenumerate(filtered_track_list):
                track = np.array([state.position for state in track.object_states])
                track = track if track.size <=10 else track[:10] # for AV track
                rot_rel_pos_track = track - focal_ped_pos_end_obs
                if apl_rotation:
                    rot_rel_pos_track = np.dot(rot_rel_pos_track, rotation_matrix)
                rot_rel_pos_track = rot_rel_pos_track[::-1] # array starts with most recent observation
                rot_rel_pos_track = rot_rel_pos_track.flatten()
                rot_rel_pos_mover_tracks[i,:rot_rel_pos_track.size] = rot_rel_pos_track

            # Extract the object types of the tracks
            object_type_list = np.array([[_MOVER_TYPE_DICT[track.object_type]] for track in filtered_track_list])

            # Create the mover matrix
            mover_matrix = np.hstack((object_type_list, rot_rel_pos_mover_tracks))

            if _DISTANCE_BASED_SELECTION:
                sorted_mover_matrix = mover_matrix[np.argsort(np.linalg.norm(mover_matrix[:,1:3], axis=1))]
                
            else:
                risk_list = np.zeros([len(filtered_track_list),1])
                focal_ped_track = scenario.tracks[int(focal_track_id)]
                pos_focal_agent = np.array(focal_ped_track.object_states[-1].position)
                vel_focal_agent = obtain_exact_velocity(focal_ped_track.object_states[-2], focal_ped_track.object_states[-1])

                for i, track in np.ndenumerate(filtered_track_list):
                    pos_other_agent = np.array(track.object_states[-1].position)
                    if len(track.object_states) > 2:
                        if len(track.object_states) > _OBSERVATION_LENGTH:
                            vel_other_agent = obtain_exact_velocity(track.object_states[_OBSERVATION_LENGTH-2], track.object_states[_OBSERVATION_LENGTH-1])
                        else:
                            vel_other_agent = obtain_exact_velocity(track.object_states[-2], track.object_states[-1])
                    else:
                        vel_other_agent = np.array(track.object_states[-1].velocity)

                    delta_alpha, ttca, dca, s_rel, d_rel = calculate_cognitive_metrics(pos_focal_agent, vel_focal_agent, pos_other_agent, vel_other_agent)
                    risk_list[i] = calculate_collision_risk(delta_alpha, ttca, dca, s_rel, d_rel)

               
                sorted_mover_matrix = mover_matrix[np.argsort(risk_list[:,0])[::-1]]

            short_sorted_mover_matrix = sorted_mover_matrix[:_MAX_NUM_AGENTS,:]

            if short_sorted_mover_matrix.shape[0] < _MAX_NUM_AGENTS:
                    short_sorted_mover_matrix = np.pad(short_sorted_mover_matrix, ((0, _MAX_NUM_AGENTS-short_sorted_mover_matrix.shape[0]), (0,0)), 'constant', constant_values=(0))

            return short_sorted_mover_matrix
        else:
            return np.zeros((_MAX_NUM_AGENTS, 21))




def create_social_matrix_and_ground_truth(scenario: ArgoverseScenario, focal_track_id: str, apl_rotation = _ROTATION):
    """
    Creates a social matrix and ground truth for a given scenario and focal track by performing the following steps:
    1. Extract the focal pedestrian track.
    2. Extract the position of the focal pedestrian at the end of the observation period.
    3. Calculate the exact heading and speed between the last two object states of the focal pedestrian.
    4. Calculate the average heading and speed between the first and second last object states of the focal pedestrian.
    5. Generate a rotation matrix to align the map with the focal pedestrian heading.
    6. Create the ground truth.
    7. Create the focal feature vector.
    8. Create the mover matrix.
    9. Concatenate the focal feature vector and mover matrix to create the social matrix.

    Args:
        scenario (scenario_serialization.ArgoverseScenario): The scenario containing the tracks.
        focal_track_id (str): The ID of the focal track.

    Global Variables:
        _OBSERVATION_LENGTH (int): The length of the observation sequence.

    Returns:
        tuple: A tuple containing the social matrix and ground truth.
            - social_matrix (numpy.ndarray): The social matrix.
            - ground_truth (numpy.ndarray): The ground truth.

    Raises:
        None
    """
    focal_ped_track = scenario.tracks[int(focal_track_id)]
    focal_ped_pos_end_obs = focal_ped_track.object_states[_OBSERVATION_LENGTH - 1].position
    focal_ped_heading_end_obs, _ = obtain_exact_heading_speed_pos(focal_ped_track.object_states[_OBSERVATION_LENGTH - 2], 
                                                        focal_ped_track.object_states[_OBSERVATION_LENGTH - 1])
    

    # rotate map to align with pedestrian heading, heading || North
    # IMPORTANT: v' = vR RIGHT SIDE MULTIPLICATION
    theta = (math.pi / 2) - focal_ped_heading_end_obs
    # | x'| _ | x | |cos(theta) sin (theta) | 
    # | y'| - | y | |-sin(theta) cos(theta) | 
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    observation, ground_truth = create_observation_and_ground_truth(focal_ped_track, focal_ped_pos_end_obs, rotation_matrix, apl_rotation)

    # obtain the feature vector of the focal pedestrian
    focal_feature_vec = np.concatenate((np.array([_MOVER_TYPE_DICT['FOCAL_PEDESTRIAN']]), observation[::-1].flatten()))

    mover_matrix = create_mover_matrix(scenario, focal_track_id, focal_ped_pos_end_obs, theta, rotation_matrix, apl_rotation)

    social_matrix = np.vstack((focal_feature_vec, mover_matrix))

    # norm = 1
    # social_matrix[:,1:] = social_matrix[:,1:] / norm

    return social_matrix, ground_truth