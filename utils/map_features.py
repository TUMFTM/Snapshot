# imports
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneSegment, LaneType
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.map.drivable_area import DrivableArea
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario, Track, ObjectType)
from av2.utils.typing import NDArrayFloat


from shapely.geometry import Polygon, box, MultiPolygon, LineString, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.affinity import translate, rotate
from typing import Tuple, List, Dict
import logging
import yaml
import numpy as np
import math
from pathlib import Path
from matplotlib.path import Path as mplPath

# local imports
from .extract_heading import obtain_exact_heading_speed_pos

# global variables
with open(Path(__file__).parent.resolve() / "../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)

_OBSERVATION_LENGTH = _CONFIG["samples"]["OBSERVATION_LENGTH"]
_OBJECT_TYPE_DICT = _CONFIG["vectorization"]["_OBJECT_TYPE_DICT"]
_MAX_NUM_VECTORS = _CONFIG["vectorization"]["MAX_NUM_VECTORS"]
_MAX_LENGTH_VECTORS = _CONFIG["vectorization"]["MAX_LENGTH_VECTORS"]
_ROTATION = _CONFIG["samples"]["ROTATION"]


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

def interpolate_points(p1, p2, max_dist):
    line = LineString([p1, p2])
    dist = line.length
    if dist <= max_dist:
        return [p1, p2]
    
    num_segments = int(math.ceil(dist / max_dist))
    points = [p1]
    
    for i in range(1, num_segments):
        fraction = i / num_segments
        interpolated_point = line.interpolate(fraction, normalized=True)
        points.append((interpolated_point.x, interpolated_point.y))
    
    points.append(p2)
    return points

def process_polygon(polygon, max_dist):
    coords = list(polygon.exterior.coords)
    new_coords = []
    
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        new_coords.extend(interpolate_points(p1, p2, max_dist))
    
    # Remove duplicates
    result = []
    for point in new_coords:
        if not result or result[-1] != point:
            result.append(point)
    
    # Ensure the polygon is closed
    if result[0] != result[-1]:
        result.append(result[0])
    
    return Polygon(result)

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

def process_vectorized_map(vectorized_map: NDArrayFloat, norm_position: NDArrayFloat, rotation_matrix: NDArrayFloat, apl_rotation: bool, apl_sorting: bool,  max_num_vectors: int) -> NDArrayFloat:
    
    vectorized_map[:, 2:] = vectorized_map[:, 2:] - np.tile(norm_position, (1, 2))

    # for visualization
    if apl_sorting:
        filtered_and_sorted_map = sorted(
            ((segment, shortest_distance_to_segment(segment)) for segment in vectorized_map),
            key=lambda x: x[1]
        )
        filtered_and_sorted_map = np.array([segment for segment, distance in filtered_and_sorted_map if not np.isnan(distance)])
        short_sorted_trans = filtered_and_sorted_map[:max_num_vectors]
    else:
        short_sorted_trans = vectorized_map[:max_num_vectors]
    
    if short_sorted_trans.shape[0] == 0:
        short_sorted_trans = np.zeros((1, 6)) 


    if apl_rotation:
        short_sorted_trans[:, 2:4] = np.dot(short_sorted_trans[:, 2:4], rotation_matrix)
        short_sorted_trans[:, 4:6] = np.dot(short_sorted_trans[:, 4:6], rotation_matrix)

    if len(short_sorted_trans) < max_num_vectors:
        short_sorted_trans = np.pad(short_sorted_trans, ((0, max_num_vectors - short_sorted_trans.shape[0]), (0,0)), 'constant', constant_values=(0))

    return short_sorted_trans

def poly_list_to_vector_light(polygon_list: List[Polygon], start_id: int, polygon_type: str):
    vectors = []
    polygon_id = start_id
    
    if polygon_list:
        for polygon in polygon_list:
            new_vectors = create_vector(polygon, polygon_type, polygon_id)
            if new_vectors is not None:
                vectors.extend(new_vectors)
                polygon_id += 1

    return vectors, polygon_id

def create_vectorized_map(map_poly_dict: Dict, ) -> NDArrayFloat:


    map_poly_dict["driv_area"] = [polygon.simplify(0.1, preserve_topology=True) for polygon in map_poly_dict["driv_area"]]
    #map_poly_dict["driv_area"] = [poly for geom in map_poly_dict["driv_area"] for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]

    map_poly_dict["lane_seg_bike"] = [polygon.simplify(0.1, preserve_topology=True) for polygon in map_poly_dict["lane_seg_bike"]]
    #map_poly_dict["lane_seg_bike"] = [poly for geom in map_poly_dict["lane_seg_bike"] for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]

    map_poly_dict["lane_seg_bus"] = [polygon.simplify(0.1, preserve_topology=True) for polygon in map_poly_dict["lane_seg_bus"]]
    #map_poly_dict["lane_seg_bus"] = [poly for geom in map_poly_dict["lane_seg_bus"] for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]

    map_poly_dict["lane_seg_vehicle"] = [polygon.simplify(0.1, preserve_topology=True) for polygon in map_poly_dict["lane_seg_vehicle"]]
    #map_poly_dict["lane_seg_vehicle"] = [poly for geom in map_poly_dict["lane_seg_vehicle"] for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])]

    for key in map_poly_dict.keys():
        map_poly_dict[key] = [process_polygon(polygon, _MAX_LENGTH_VECTORS) for polygon in map_poly_dict[key]]



    vectors = []
    polygon_id = 1


    # vectorize drivable_areas
    new_vectors, polygon_id = poly_list_to_vector_light(map_poly_dict["driv_area"], polygon_id, "drivable_areas")
    vectors.extend(new_vectors)

    # vectorize pedestrian_crossings
    new_vectors, polygon_id = poly_list_to_vector_light(map_poly_dict["ped_cross"], polygon_id, "pedestrian_crossings")
    vectors.extend(new_vectors)

    # # vectorize lane_segments BIKE
    # new_vectors, polygon_id = poly_list_to_vector_light(map_poly_dict["lane_seg_bike"], polygon_id, "LaneType_BIKE")
    # vectors.extend(new_vectors)

    # # vectorize lane_segments BUS
    # new_vectors, polygon_id = poly_list_to_vector_light(map_poly_dict["lane_seg_bus"], polygon_id, "LaneType_BUS")
    # vectors.extend(new_vectors)

    # # vectorize lane_segments VEHICLE
    # new_vectors, polygon_id = poly_list_to_vector_light(map_poly_dict["lane_seg_vehicle"], polygon_id, "LaneType_VEHICLE")
    # vectors.extend(new_vectors)
    
    
    max_id = polygon_id - 1
    vectorized_map = np.array(vectors)

    # Normalize id
    if max_id > 0:
        vectorized_map[:, 1] = 2 * (vectorized_map[:, 1] / max_id) - 1

    # avoid vectors being empty
    if vectorized_map.shape[0] == 0:
        vectorized_map = np.zeros((1, 6)) 

    return vectorized_map

def map_file_to_poly_dict(map_file: ArgoverseStaticMap):
    drivable_areas: List[DrivableArea] = list(map_file.vector_drivable_areas.values())
    pedestrian_crossings: List[PedestrianCrossing] = list(map_file.vector_pedestrian_crossings.values())
    lane_segments: List[LaneSegment] = list(map_file.vector_lane_segments.values())

    # vectorize drivable_areas
    driv_area = [make_valid(Polygon(dav.xyz[:,:2])) for dav in drivable_areas]
    driv_area = [sub_geom for geom in driv_area for sub_geom in 
                 (geom.geoms if isinstance(geom, GeometryCollection) else [geom]) if isinstance(sub_geom, Polygon)]
    
    # controll if all polygons are counter clockwise
    for i, polygon in enumerate(driv_area):
        if not polygon.exterior.is_ccw:
            driv_area[i] = Polygon(list(polygon.exterior.coords)[::-1])


    # vectorize pedestrian_crossings
    ped_cross = [make_valid(Polygon(polygon.polygon[:,:2])) for polygon in pedestrian_crossings]
    ped_cross = [sub_geom for geom in ped_cross for sub_geom in 
                 (geom.geoms if isinstance(geom, GeometryCollection) else [geom]) if isinstance(sub_geom, Polygon)]
    for i, polygon in enumerate(ped_cross):
        if not polygon.exterior.is_ccw:
            ped_cross[i] = Polygon(list(polygon.exterior.coords)[::-1])



    # vectorize lane_segments BIKE
    lane_seg_bike = [make_valid(Polygon(polygon.polygon_boundary[:,:2])) for polygon in lane_segments if polygon.lane_type == LaneType.BIKE]
    lane_seg_bike = [sub_geom for geom in lane_seg_bike for sub_geom in 
                 (geom.geoms if isinstance(geom, GeometryCollection) else [geom]) if isinstance(sub_geom, Polygon)]
    for i, polygon in enumerate(lane_seg_bike):
        if not polygon.exterior.is_ccw:
            lane_seg_bike[i] = Polygon(list(polygon.exterior.coords)[::-1])


    # vectorize lane_segments BUS
    lane_seg_bus = [make_valid(Polygon(polygon.polygon_boundary[:,:2])) for polygon in lane_segments if polygon.lane_type == LaneType.BUS]
    lane_seg_bus = [sub_geom for geom in lane_seg_bus for sub_geom in 
                 (geom.geoms if isinstance(geom, GeometryCollection) else [geom]) if isinstance(sub_geom, Polygon)]
    for i, polygon in enumerate(lane_seg_bus):
        if not polygon.exterior.is_ccw:
            lane_seg_bus[i] = Polygon(list(polygon.exterior.coords)[::-1])



    # vectorize lane_segments VEHICLE
    lane_seg_vehicle = [make_valid(Polygon(polygon.polygon_boundary[:,:2])) for polygon in lane_segments if polygon.lane_type == LaneType.VEHICLE]
    lane_seg_vehicle = [sub_geom for geom in lane_seg_vehicle for sub_geom in 
                 (geom.geoms if isinstance(geom, GeometryCollection) else [geom]) if isinstance(sub_geom, Polygon)]
    for i, polygon in enumerate(lane_seg_vehicle):
        if not polygon.exterior.is_ccw:
            lane_seg_vehicle[i] = Polygon(list(polygon.exterior.coords)[::-1])


    map_poly_dict = {"driv_area": driv_area, "ped_cross": ped_cross, "lane_seg_bike": lane_seg_bike, "lane_seg_bus": lane_seg_bus, "lane_seg_vehicle": lane_seg_vehicle}
    return map_poly_dict

def vectorize_map(avm: ArgoverseStaticMap, scenario: ArgoverseScenario, focal_track_id: str, current_logger: logging.Logger, apl_rotation = _ROTATION, apl_sorting = True, max_num_vectors = _MAX_NUM_VECTORS):
    '''
    Vectorizes the map based on the focal pedestrian track

    Args:
        avm (ArgoverseStaticMap): The ArgoverseStaticMap object.
        scenario (ArgoverseScenario): The ArgoverseScenario object.
        focal_track_id (str): The ID of the focal pedestrian track.
        current_logger (logging.Logger): The logger object.
        apl_rotation (bool): Whether to apply rotation.

    Returns:
        NDArrayFloat: The vectorized map.
        np.ndarray: The map corners.
        np.ndarray: The raster map.

    Raises:
        None
    '''
    
    # Get pedestrian track
    focal_ped_track = scenario.tracks[int(focal_track_id)]
    focal_ped_position_end_obs = np.array(focal_ped_track.object_states[_OBSERVATION_LENGTH - 1].position)
    focal_ped_heading_end_obs, _ = obtain_exact_heading_speed_pos(focal_ped_track.object_states[_OBSERVATION_LENGTH - 2], 
                                                        focal_ped_track.object_states[_OBSERVATION_LENGTH - 1])
    
    # rotate map to align with pedestrian heading, heading || North

    theta = (math.pi / 2) - focal_ped_heading_end_obs
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])


    try:
        map_poly_dict = map_file_to_poly_dict(avm)
    except Exception as e:
        current_logger.error(f"Error in map_file_to_poly_dict: {e}")

    try:
        vectorized_map = create_vectorized_map(map_poly_dict)
    except Exception as e:
        current_logger.error(f"Error in create_vectorized_map: {e}")

    try:
        vectorized_map = process_vectorized_map(vectorized_map, focal_ped_position_end_obs, rotation_matrix, apl_rotation, apl_sorting, max_num_vectors)
    except Exception as e:
        current_logger.error(f"Error in process_vectorized_map: {e}")


    # Normalisation
    # norm = 1
    # vectorized_map[:, 2:6] = vectorized_map[:, 2:6] / norm


    return vectorized_map