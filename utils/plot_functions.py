# imports
import numpy as np
import math
from matplotlib.patches import Rectangle, Circle
import matplotlib.colors as mcolors

# global variables
grey  = "#9090A0"
orange  = "#E37227"
color_dict = {
    0.6: grey, # car dark_blue
    1.0: grey, # bus
    0.2: grey, # motorbike
    -0.2: grey, # cyclists dark_orange
    -0.6: grey, # ped
    -1: orange # focal 
}

shape_dict = {
    0.6: [0.2, 0.1], # car [4m,2m]
    1.0: [0.5, 0.15], # bus
    0.2: [0.05, 0.02], # motorbike
    -0.2: [0.05, 0.02], # cyclists
}

def plot_agent(ax, agent):
    """
    Plot the agent's current position using a shape that corresponds to its type.
    - Vehicles (cars, buses, cyclists, motorbikes) are drawn as rectangles.
    - Pedestrians are drawn as circles.
    """

    agent[1:] = agent[1:] / 20 # norm
    agent_type = agent[0]
    position = agent[1:3]
    delta_x = agent[1] - agent[15]   # calculate over larger distance to compensate noise
    delta_y = agent[2] - agent[16]
    heading = math.atan2(delta_y, delta_x)
    
    if agent_type in [-0.2, 0.2, 0.6, 1.0]:
        # rectangles
        height, width = shape_dict[agent_type]
        d = np.hypot(height, width)
        theta_2 = math.atan2(width, height)
        pivot_x = position[0] - (d / 2) * math.cos(heading + theta_2)
        pivot_y = position[1] - (d / 2) * math.sin(heading + theta_2)

        base_color = color_dict[agent_type]
        fill_color = mcolors.to_rgba(base_color, alpha=0.3)
        edge_color = mcolors.to_rgba(base_color, alpha=1.0)

        vehicle_bounding_box = Rectangle(
            (pivot_x, pivot_y),
            height,
            width,
            angle=np.degrees(heading),
            facecolor=fill_color,
            edgecolor=edge_color,
            lw=2
        )
        ax.add_patch(vehicle_bounding_box)
    else:
        # Draw a circle with fixed radius for pedestrians.

        base_color = color_dict[agent_type]
        if agent_type == -1.0:
            fill_color = mcolors.to_rgba(base_color, alpha=0.4)
        else:
            fill_color = mcolors.to_rgba(base_color, alpha=0.4) 
        edge_color = mcolors.to_rgba(base_color, alpha=1.0) 

        circ = Circle(
            position,
            0.02,
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=1.0,
            # zorder=100
            # hatch=hatch_dict.get(agent_type, "")
        )
        ax.add_patch(circ)

    # plot motion history
    if agent_type != -1.0:
        ax.scatter(agent[1::2], agent[2::2], color=color_dict[agent_type], s=1.5, zorder=2)
    else:
        ax.scatter(agent[1::2], agent[2::2], color=color_dict[agent_type], s=0.5, zorder=2)


def plot_vec_map(ax, map_matrix):
    map_matrix[:,2:] = map_matrix[:,2:]  / 20
    driv_area = map_matrix[map_matrix[:,0] == -0.9][:,1:]
    ped_crossing = map_matrix[map_matrix[:,0] == 0.3][:,1:]
    lanes = map_matrix[map_matrix[:,0] == -0.3][:,1:]

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

    if driv_area.size != 0:
        key = driv_area[0][0]
        current_matrix = []
        for row in driv_area:
            if row[0] == key:
                current_matrix.append(row)
            else:
                current_matrix = np.array(current_matrix)
                ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="#66A1E0", linewidth=0.5)
                ax.fill(current_matrix[:,1], current_matrix[:,2], color="#66A1E0", alpha=0.2)
                key = row[0]
                current_matrix = [row]
        current_matrix = np.array(current_matrix)
        ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="#66A1E0", linewidth=0.5)
        ax.fill(current_matrix[:,1], current_matrix[:,2], color="#66A1E0", alpha=0.2)

    if ped_crossing.size != 0:
        key = ped_crossing[0][0]
        current_matrix = []
        for row in ped_crossing:
            if row[0] == key:
                current_matrix.append(row)
            else:
                current_matrix = np.array(current_matrix)
                ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="#F18C3B", linewidth=0.5)
                ax.fill(current_matrix[:,1], current_matrix[:,2], color="#F18C3B", alpha=0.2)
                key = row[0]
                current_matrix = [row]
        current_matrix = np.array(current_matrix)
        ax.plot([current_matrix[:,1], current_matrix[:,3]], [current_matrix[:,2], current_matrix[:,4]], color="#F18C3B", linewidth=0.5)
        ax.fill(current_matrix[:,1], current_matrix[:,2], color="#F18C3B", alpha=0.2)
