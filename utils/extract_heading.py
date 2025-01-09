# imports
import numpy as np
import math
from typing import Tuple, List
from av2.datasets.motion_forecasting.data_schema import ObjectState


def obtain_exact_velocity(state_start: ObjectState, state_end: ObjectState) -> Tuple[float, float]:
    if state_end.timestep > state_start.timestep:
        dt = (state_end.timestep - state_start.timestep)/10
        delta_x = state_end.position[0] - state_start.position[0]
        delta_y = state_end.position[1] - state_start.position[1]
        velocity = np.array([delta_x / dt, delta_y / dt])
        return velocity
    else:
        raise ValueError(f"Invalid timesteps: state_end.timestep ({state_end.timestep}) <= state_start.timestep ({state_start.timestep})")


def obtain_exact_heading_speed_pos(state_start: ObjectState, state_end: ObjectState) -> Tuple[float, float]:
    """
    Calculates the exact heading and speed between two object states by performing the following steps:
    1. Calculate the time difference between the two states.
    2. Calculate the change in x and y positions.
    3. Calculate the heading using the arctangent of the change in y and x positions.
    4. Calculate the speed by dividing the change in x and y positions by the time 
        difference and taking the hypotenuse of the resulting x and y speeds.

    Args:
        state_start (ObjectState): The starting state of the object.
        state_end (ObjectState): The ending state of the object.

    Global Variables:
        None

    Returns:
        Tuple[float, float]: A tuple containing the exact heading and speed between the two states.
        Heading is in radians between [-π, π]. 0 is along the positive x-axis.
        Speed is in m/s.

    Raises:
        ValueError: If the ending state has a timestep that is less than or equal to the starting state's timestep.
    """
    if state_end.timestep > state_start.timestep:
        # Calculate the heading and speed
        dt = (state_end.timestep - state_start.timestep) / 10
        delta_x = state_end.position[0] - state_start.position[0]
        delta_y = state_end.position[1] - state_start.position[1]
        # atan2: The angle θ in radians between the positive x-axis of a plane and the 
        # point given by the coordinates (x, y). The result is in the range [-π, π].
        heading = math.atan2(delta_y, delta_x)
        speed = np.hypot(delta_x / dt, delta_y / dt)
        return heading, speed
    else:
        raise ValueError(f"Invalid timesteps: state_end.timestep ({state_end.timestep}) <= state_start.timestep ({state_start.timestep})")



def obtain_current_heading_speed_from_data(state_end: ObjectState) -> Tuple[float, float]:
    """
    Obtain the current heading and speed from the given ObjectState.

    Args:
        state_end (ObjectState): The ObjectState object containing the state information.

    Global Variables:
        None

    Returns:
    Tuple[float, float]: A tuple containing the current heading and speed.

    Raises:
        None
    """
    current_heading = state_end.heading
    current_speed = np.linalg.norm(state_end.velocity)
    return current_heading, current_speed



def obtain_avg_heading_speed_from_data(all_object_states: List[ObjectState]) -> Tuple[float, float]:
    """
    Calculate the average heading and speed from a list of object states.

    Args:
        all_object_states (List[ObjectState]): A list of object states.

    Global Variables:
        None

    Returns:
        Tuple[float, float]: A tuple containing the average heading and average speed.

    Raises:
        None
    """
    avg_heading = np.mean([elmt.heading for elmt in all_object_states])
    avg_speed = np.mean([np.linalg.norm(elmt.velocity) for elmt in all_object_states])
    return avg_heading, avg_speed