# imports
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneSegment, LaneType
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.map.drivable_area import DrivableArea
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario, Track, ObjectType)
from av2.utils.typing import NDArrayFloat

from shapely.geometry import Polygon, box, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
from typing import Tuple, List
import logging
import yaml
import numpy as np
import math

# local imports
from .extract_heading import obtain_exact_heading_speed_pos

# global variables
with open("config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

_OBSERVATION_LENGTH = _CONFIG["samples"]["OBSERVATION_LENGTH"]
_OBJECT_TYPE_DICT = _CONFIG["vectorization"]["_OBJECT_TYPE_DICT"]
_MAX_NUM_VECTORS = _CONFIG["vectorization"]["MAX_NUM_VECTORS"]


def shortest_distance_to_segment(vector: NDArrayFloat) -> float:
    """
    Calculates the shortest distance between a given vector and the origin

    Args:
        vector (numpy.ndarray): Transformed ployline with respect to the focal agent at (0,0)

    Returns:
        float: shortest ditance to the origin

    Raises:
        None
    """
    A = np.array([0,0])
    B = vector[2:4]
    C = vector[4:]

    BA = A - B
    BC = C - B
    t = np.dot(BA, BC) / (np.dot(BC, BC) +0.00001)
    
    if 0 <= t <= 1:
        closest_point = B + t * BC
        return np.linalg.norm(A - closest_point)
    elif t < 0:
        return np.linalg.norm(BA)
    else:
        return np.linalg.norm(A - C)


def vectorize_static_obstacles(position_xy: NDArrayFloat, ped_pos: Tuple[float,float], rotation_matrix: NDArrayFloat, radius: int) -> NDArrayFloat:
    """
    Vectorizes a polygon by performing the following steps:
    1. Subtract pedestrian position from all points in the polygon.
    2. Compute the L1 norm and filter points based on a given radius.
    3. Rotate the filtered points using a rotation matrix.
    4. Create a vector for each segment of the polygon.

    Args:
        position_xy (numpy.ndarray): Array of shape (N, 2) representing the x and y coordinates of the points.
        ped_pos (tuple): Tuple of two floats representing the x and y coordinates of the pedestrian.
        rotation_matrix (numpy.ndarray): 2x2 rotation matrix used to rotate the points.
        radius (int): Maximum L1 norm distance from the pedestrian for a point to be included in the result.

    Global Variables:
        _OBJECT_TYPE_DICT (dict): Dictionary mapping object types to their corresponding values.

    Returns:
        np.ndarray: The vectorized representation of the polygon. Shape (M, 6) where M is the number of vectors. Each vector is represented by a row with the following columns:
            - type (float): Type of the vector. -0.9 for drivable areas, -0.3 for lane segments, 0.3 for pedestrian crossings, 0.9 for static obstacles.
            - id (float): always 0 for static obstacles.
            - x1 (float): x-coordinate of the obstacles.
            - y1 (float): y-coordinate of the obstacles.
            - x2 (float): x-coordinate of the obstacles.
            - y2 (float): y-coordinate of the obstacles.

    Raises:
        None
    """
    new_vectors = None
    # Subtract pedestrian position from all points
    relative_positions_xy = position_xy - ped_pos

    # Compute L1 norm and filter points 
    l1_norm = np.linalg.norm(relative_positions_xy, ord=1, axis=1)
    filtered_points_xy = relative_positions_xy[l1_norm < radius]

    if len(filtered_points_xy) > 1:
        # Rotate filtered points
        start_points_xy = np.dot(filtered_points_xy, rotation_matrix)

        # Create vector for each segment
        num_vectors = len(start_points_xy)
        types = np.full(num_vectors, _OBJECT_TYPE_DICT["static_obstacles"])
        new_vectors = np.column_stack((types, np.zeros((num_vectors,1)), start_points_xy, start_points_xy))

    return new_vectors

def create_vector(polygon: Polygon, object_type, id: int) -> NDArrayFloat:
    """
    Create a vector representation of a polygon.

    Args:
        polygon (Polygon): The polygon object.
        object_type: The type of the object.
        id (int): The ID of the object.

    Global Variables:
        _OBJECT_TYPE_DICT (dict): Dictionary mapping object types to their corresponding values.

    Returns:
        NDArrayFloat: The vector representation of the polygon.

    Raises:
        None
    """
    start_points_xy = np.column_stack(polygon.exterior.xy)
    end_points_xy = np.roll(start_points_xy, -1, axis=0)
    num_vectors = len(start_points_xy)
    id_vector = [id]*num_vectors
    types = np.full(num_vectors, _OBJECT_TYPE_DICT[object_type])
    new_vectors = np.column_stack((types, id_vector, start_points_xy, end_points_xy))

    return new_vectors


def vectorize_map(avm: ArgoverseStaticMap, scenario: ArgoverseScenario, focal_track_id: str, logger, radius = None) -> NDArrayFloat:
    """
    Vectorizes the map using Shapely library.

    Args:
        avm (ArgoverseStaticMap): The ArgoverseStaticMap object containing map semantics.
        scenario (ArgoverseScenario): The ArgoverseScenario object containing scenario information.
        focal_track_id (str): The ID of the focal pedestrian track.

    Global Variables:
        _CONFIG (dict): Dictionary containing configuration parameters.
        _OBSERVATION_LENGTH (int): The length of the observation sequence.

    Returns:
        NDArrayFloat: The vectorized map as a numpy array.

    Raises:
        None
    """
    # Get pedestrian track
    focal_ped_track = scenario.tracks[int(focal_track_id)]
    focal_ped_position_end_obs = focal_ped_track.object_states[_OBSERVATION_LENGTH - 1].position
    focal_ped_heading_end_obs, _ = obtain_exact_heading_speed_pos(focal_ped_track.object_states[_OBSERVATION_LENGTH - 2], 
                                                        focal_ped_track.object_states[_OBSERVATION_LENGTH - 1])
    
    # rotate map to align with pedestrian heading, heading || North
    # IMPORTANT: v = vR RIGHT SIDE MULTIPLICATION
    theta = (math.pi / 2) - focal_ped_heading_end_obs
    # | x'| _ | x | |cos(theta) sin (theta) | 
    # | y'| - | y | |-sin(theta) cos(theta) | 
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    # extract polygons of map semantics
    drivable_areas: List[DrivableArea] = list(avm.vector_drivable_areas.values())
    pedestrian_crossings: List[PedestrianCrossing] = list(avm.vector_pedestrian_crossings.values())
    lane_segments: List[LaneSegment] = list(avm.vector_lane_segments.values())

    vectors = []
    polygon_id = 1

    # Define cut boxes for each object type
    if radius is None:
        cut_box_driv_area = box(-_CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"], -_CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"], _CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"], _CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"])
        cut_box_ped_cros = box(-_CONFIG["vectorization"]["RADIUS_PED_CROSS"], -_CONFIG["vectorization"]["RADIUS_PED_CROSS"], _CONFIG["vectorization"]["RADIUS_PED_CROSS"], _CONFIG["vectorization"]["RADIUS_PED_CROSS"])
        cut_box_lane_segm = box(-_CONFIG["vectorization"]["RADIUS_LANE_SEG"], -_CONFIG["vectorization"]["RADIUS_LANE_SEG"], _CONFIG["vectorization"]["RADIUS_LANE_SEG"], _CONFIG["vectorization"]["RADIUS_LANE_SEG"])
        rad_static_obst = _CONFIG["vectorization"]["RADIUS_STATIC_OBST"]
        radius_norm = max(_CONFIG["vectorization"]["RADIUS_DRIVABLE_AREA"], _CONFIG["vectorization"]["RADIUS_PED_CROSS"], _CONFIG["vectorization"]["RADIUS_LANE_SEG"])
    else:
        cut_box_driv_area = box(-radius, -radius, radius, radius)
        cut_box_ped_cros = box(-radius, -radius, radius, radius)
        cut_box_lane_segm = box(-radius, -radius, radius, radius)
        rad_static_obst = radius
        radius_norm = radius


    # vectorize drivable_areas
    clipped_polygon = [make_valid(Polygon(np.dot(dav.xyz[:,:2]-focal_ped_position_end_obs, rotation_matrix))).intersection(cut_box_driv_area) for dav in drivable_areas]
    clipped_polygon = [poly for geom in clipped_polygon for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]
    clipped_polygon = unary_union(clipped_polygon)
    clipped_polygon = [poly for geom in [clipped_polygon] for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]
    clipped_polygon = [poly for poly in clipped_polygon if not poly.is_empty and isinstance(poly, Polygon)]

    if clipped_polygon:
        total_points = sum(len(polygon.exterior.coords) for polygon in clipped_polygon)
        simplified_polygon = clipped_polygon
        tolerance = 0.1
        while total_points > 95:
                logging.debug(f"Reducing points in drivable areas at {avm.log_id} with tolerance {tolerance}")
                simplified_polygon = [polygon.simplify(tolerance, preserve_topology=True) for polygon in clipped_polygon]
                total_points = sum(len(polygon.exterior.coords) for polygon in simplified_polygon)
                tolerance += 0.1

        for polygon in simplified_polygon:
            new_vectors = create_vector(polygon, "drivable_areas", polygon_id)
            if new_vectors is not None:
                vectors.extend(new_vectors)
                polygon_id += 1



    # vectorize pedestrian_crossings
    clipped_polygon = [make_valid(Polygon(np.dot(polygon.polygon[:,:2]-focal_ped_position_end_obs, rotation_matrix))).intersection(cut_box_ped_cros) for polygon in pedestrian_crossings]
    clipped_polygon = [poly for geom in clipped_polygon for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]
    clipped_polygon = [poly for poly in clipped_polygon if not poly.is_empty and isinstance(poly, Polygon)]
    if clipped_polygon:
        for polygon in clipped_polygon:
            new_vectors = create_vector(polygon, "pedestrian_crossings", polygon_id)
            if new_vectors is not None:
                vectors.extend(new_vectors)
                polygon_id += 1


    # vectorize lane_segments BIKE
    clipped_polygon = [make_valid(Polygon(np.dot(polygon.polygon_boundary[:,:2]-focal_ped_position_end_obs, rotation_matrix))).intersection(cut_box_lane_segm) for polygon in lane_segments if polygon.lane_type == LaneType.BIKE]
    clipped_polygon = [poly for geom in clipped_polygon for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]
    clipped_polygon = [poly for poly in clipped_polygon if not poly.is_empty and isinstance(poly, Polygon)]
    if clipped_polygon:
        simplified_polygon = [polygon.simplify(0.1, preserve_topology=True) for polygon in clipped_polygon]
        for polygon in simplified_polygon:
            new_vectors = create_vector(polygon, "LaneType_BIKE", polygon_id)
            if new_vectors is not None:
                vectors.extend(new_vectors)
                polygon_id += 1

    # vectorize lane_segments BUS
    clipped_polygon = [make_valid(Polygon(np.dot(polygon.polygon_boundary[:,:2]-focal_ped_position_end_obs, rotation_matrix))).intersection(cut_box_lane_segm) for polygon in lane_segments if polygon.lane_type == LaneType.BUS]
    clipped_polygon = [poly for geom in clipped_polygon for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]
    clipped_polygon = [poly for poly in clipped_polygon if not poly.is_empty and isinstance(poly, Polygon)]
    if clipped_polygon:
        simplified_polygon = [polygon.simplify(0.3, preserve_topology=True) for polygon in clipped_polygon]
        for polygon in simplified_polygon:
            new_vectors = create_vector(polygon, "LaneType_BUS", polygon_id)
            if new_vectors is not None:
                vectors.extend(new_vectors)
                polygon_id += 1

    # vectorize lane_segments VEHICLE
    clipped_polygon = [make_valid(Polygon(np.dot(polygon.polygon_boundary[:,:2]-focal_ped_position_end_obs, rotation_matrix))).intersection(cut_box_lane_segm) for polygon in lane_segments if polygon.lane_type == LaneType.VEHICLE]
    clipped_polygon = [poly for geom in clipped_polygon for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]
    clipped_polygon = [poly for poly in clipped_polygon if not poly.is_empty and isinstance(poly, Polygon)]
    if clipped_polygon:
        simplified_polygon = [polygon.simplify(0.5, preserve_topology=True) for polygon in clipped_polygon]
        for polygon in simplified_polygon:
            new_vectors = create_vector(polygon, "LaneType_VEHICLE", polygon_id)
            if new_vectors is not None:
                vectors.extend(new_vectors)
                polygon_id += 1



    max_id = polygon_id

    # vectorize static obstacles
    # static_pos_array = [[track.object_states[0].position[0], track.object_states[0].position[1]] 
    #                     for track in scenario.tracks if track.object_type 
    #                     in {ObjectType.STATIC, ObjectType.BACKGROUND, ObjectType.CONSTRUCTION, ObjectType.RIDERLESS_BICYCLE, ObjectType.UNKNOWN}]
    # if len(static_pos_array) > 0:
    #     new_vectors = vectorize_static_obstacles(np.array(static_pos_array), focal_ped_position_end_obs, rotation_matrix, rad_static_obst)
    #     if new_vectors is not None:
    #         vectors.extend(new_vectors)

    # sort vectors by distance
    vectors = sorted(vectors, key=shortest_distance_to_segment)
    vectorized_map = np.array(vectors)

    # Normalize vectors
    if vectorized_map.shape[0] > 0:
        vectorized_map[:, 1] = 2 * (vectorized_map[:, 1] / max_id) - 1
        vectorized_map[:, 2:6] = vectorized_map[:, 2:6] / radius_norm
    else:
        vectorized_map = np.zeros((1, 6)) # avoid vectors being empty
        logger.debug(f"No vectors found in the map: {avm.log_id}")


    if vectorized_map.shape[0] < _MAX_NUM_VECTORS:
        vectorized_map = np.pad(vectorized_map, ((0, _MAX_NUM_VECTORS - vectorized_map.shape[0]), (0,0)), 'constant', constant_values=(0))
    else:
        logger.debug(f"Map matrix for scenario {scenario.scenario_id} has more than {str(_MAX_NUM_VECTORS)} vectors")
        vectorized_map = vectorized_map[:_MAX_NUM_VECTORS]

    return vectorized_map