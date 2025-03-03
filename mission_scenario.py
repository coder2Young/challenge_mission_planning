#!/usr/bin/env python3

# Copyright 2024 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Mission planning and execution for a single drone with optimization."""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2024 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
from time import sleep
import time
import yaml
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search

from as2_python_api.drone_interface import DroneInterface
try:
    from challenge_mission_planning.drone_camera import ArucoDetectorDrone
except ImportError:
    # Fallback if package is not installed
    from drone_camera import ArucoDetectorDrone

import rclpy

# Mission parameters
TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0  # Max speed in m/s
SLEEP_TIME = 0.5  # Sleep time between behaviors in seconds
SPEED = 1.0  # Max speed in m/s
LAND_SPEED = 0.5  # Max speed in m/s
SCAN_DURATION = 5.0  # Duration to scan for ArUco markers at each waypoint (increased from 2.0)
OBSTACLE_SAFETY_MARGIN = 0.5  # Safety margin around obstacles in meters
WAYPOINT_TOLERANCE = 0.1  # Distance tolerance for waypoint achievement in meters

# Path planning methods
PATH_PLANNING_METHODS = ['direct', 'a_star', 'rrt']
DEFAULT_PATH_PLANNING = 'a_star'

# TSP solver methods
TSP_METHODS = ['dynamic_programming', 'simulated_annealing', 'local_search']
DEFAULT_TSP_METHOD = 'simulated_annealing'

# Default parameters
DEFAULT_TSP_MATRIX_METHOD = 'euclidean'  # Default TSP matrix calculation method: 'euclidean' or 'pathplanning'


def drone_start(drone_interface: DroneInterface, logger=None) -> bool:
    """
    Take off the drone.

    :param drone_interface: DroneInterface object
    :param logger: Optional logger for mission logging
    :return: Bool indicating if the take off was successful
    """
    print('Start mission')
    if logger:
        logger.info('Starting mission')

    # Arm
    print('Arm')
    success = drone_interface.arm()
    print(f'Arm success: {success}')
    if logger:
        logger.info(f'Arm success: {success}')

    # Offboard
    print('Offboard')
    success = drone_interface.offboard()
    print(f'Offboard success: {success}')
    if logger:
        logger.info(f'Offboard success: {success}')

    # Take Off
    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'Take Off success: {success}')
    if logger:
        logger.info(f'Take Off success: {success}')

    return success


def is_collision_free(start, end, obstacles, margin=OBSTACLE_SAFETY_MARGIN):
    """
    Check if a straight line path between start and end collides with any obstacle.
    
    :param start: 3D start point [x, y, z]
    :param end: 3D end point [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :return: True if path is collision-free, False otherwise
    """
    if not obstacles:
        return True
    
    # Convert to numpy arrays for easier manipulation
    start = np.array(start)
    end = np.array(end)
    path_vector = end - start
    path_length = np.linalg.norm(path_vector)
    path_direction = path_vector / path_length
    
    # Check collision by sampling points along the path
    num_samples = int(path_length * 10)  # 10 samples per meter
    num_samples = max(10, num_samples)  # At least 10 samples
    
    for i in range(num_samples + 1):
        t = i / num_samples
        point = start + t * path_vector
        
        # Check if point is inside any obstacle plus margin
        for obs_id, obs in obstacles.items():
            obs_pos = np.array([obs['x'], obs['y'], obs['z']])
            obs_size = np.array([obs['w'], obs['d'], obs['h']]) / 2 + margin
            
            # Check if point is inside obstacle's bounding box with margin
            if (abs(point[0] - obs_pos[0]) < obs_size[0] and
                abs(point[1] - obs_pos[1]) < obs_size[1] and
                abs(point[2] - obs_pos[2]) < obs_size[2]):
                return False
    
    return True


def a_star_path_planning(start, goal, obstacles, margin=OBSTACLE_SAFETY_MARGIN):
    """
    Implement A* algorithm for path planning between viewpoints.
    
    :param start: 3D start point [x, y, z]
    :param goal: 3D end point [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :return: List of waypoints including start and goal
    """
    # Check if direct path is possible
    if is_collision_free(start, goal, obstacles, margin):
        return [start, goal]
    
    # Create a 3D grid for path planning
    grid_resolution = 1.0  # 1 meter resolution
    
    # Determine grid bounds with padding
    bounds_min = [-15, -15, 0]
    bounds_max = [15, 15, 10]
    
    # Create graph for A*
    G = nx.Graph()
    
    # Create grid points
    grid_points = []
    for x in np.arange(bounds_min[0], bounds_max[0], grid_resolution):
        for y in np.arange(bounds_min[1], bounds_max[1], grid_resolution):
            for z in np.arange(bounds_min[2], bounds_max[2], grid_resolution):
                point = (x, y, z)
                # Add point if it's collision-free
                is_valid = True
                for obs_id, obs in obstacles.items():
                    obs_pos = np.array([obs['x'], obs['y'], obs['z']])
                    obs_size = np.array([obs['w'], obs['d'], obs['h']]) / 2 + margin
                    
                    if (abs(point[0] - obs_pos[0]) < obs_size[0] and
                        abs(point[1] - obs_pos[1]) < obs_size[1] and
                        abs(point[2] - obs_pos[2]) < obs_size[2]):
                        is_valid = False
                        break
                
                if is_valid:
                    grid_points.append(point)
                    G.add_node(point)
    
    # Add start and goal to the graph
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    G.add_node(start_tuple)
    G.add_node(goal_tuple)
    grid_points.append(start_tuple)
    grid_points.append(goal_tuple)
    
    # Connect neighboring points
    for i, p1 in enumerate(grid_points):
        for p2 in grid_points[i+1:]:
            # Connect if direct path is possible and within reasonable distance
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < grid_resolution * 1.8 and is_collision_free(p1, p2, obstacles, margin):
                G.add_edge(p1, p2, weight=dist)
    
    # A* search
    try:
        path = nx.astar_path(G, start_tuple, goal_tuple, heuristic=lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))
        path = [list(p) for p in path]
        return path
    except nx.NetworkXNoPath:
        # If no path found, try with direct connection
        print("A* could not find a path, returning direct connection")
        return [start, goal]


def calculate_tsp_matrix(viewpoints, obstacles, matrix_method=DEFAULT_TSP_MATRIX_METHOD, path_planning_method=DEFAULT_PATH_PLANNING):
    """
    Calculate distance matrix for TSP optimization.
    
    :param viewpoints: Dictionary of viewpoints with position information
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param matrix_method: Method to calculate TSP matrix distances: 'euclidean' or 'pathplanning'
    :param path_planning_method: Method to use for path planning if matrix_method='pathplanning'
    :return: 2D numpy array containing distances between all viewpoints
    """
    print(f"Starting TSP matrix calculation using method: {matrix_method}")
    matrix_start_time = time.time()
    
    viewpoint_ids = list(viewpoints.keys())
    n = len(viewpoint_ids)
    distance_matrix = np.zeros((n, n))
    
    # Calculate distances between all pairs of viewpoints
    for i in range(n):
        vp_i = viewpoints[viewpoint_ids[i]]
        pos_i = [vp_i["x"], vp_i["y"], vp_i["z"]]
        
        for j in range(i+1, n):
            vp_j = viewpoints[viewpoint_ids[j]]
            pos_j = [vp_j["x"], vp_j["y"], vp_j["z"]]
            
            # Calculate the distance based on the selected method
            if matrix_method == 'euclidean':
                # Simple Euclidean distance without considering obstacles
                dist = np.linalg.norm(np.array(pos_j) - np.array(pos_i))
            elif matrix_method == 'pathplanning':
                # Use path planning to calculate distances considering obstacles
                if is_collision_free(pos_i, pos_j, obstacles):
                    # Direct distance if path is collision-free
                    dist = np.linalg.norm(np.array(pos_j) - np.array(pos_i))
                else:
                    # Path planning distance
                    path = plan_path_between_viewpoints(pos_i, pos_j, obstacles, method=path_planning_method)
                    if path:
                        # Calculate path length
                        dist = 0
                        for k in range(len(path) - 1):
                            leg_distance = np.linalg.norm(np.array(path[k+1]) - np.array(path[k]))
                            dist += leg_distance
                    else:
                        # If no path found, use a large value
                        dist = 1000.0
            else:
                # Default to Euclidean distance
                print(f"Warning: Unknown matrix method '{matrix_method}', defaulting to Euclidean distance")
                dist = np.linalg.norm(np.array(pos_j) - np.array(pos_i))
            
            # Update distance matrix
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    matrix_end_time = time.time()
    matrix_calculation_time = matrix_end_time - matrix_start_time
    print(f"TSP matrix calculation completed in {matrix_calculation_time:.2f} seconds.")
    
    return distance_matrix


def optimize_viewpoint_order(viewpoints, obstacles, tsp_method=DEFAULT_TSP_METHOD, 
                           matrix_method=DEFAULT_TSP_MATRIX_METHOD, path_planning_method=DEFAULT_PATH_PLANNING):
    """
    Optimize the order of viewpoints using TSP solver.
    
    :param viewpoints: Dictionary of viewpoints with position information
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param tsp_method: Method to use for TSP solving
    :param matrix_method: Method to calculate TSP matrix distances: 'euclidean' or 'pathplanning'
    :param path_planning_method: Method to use for path planning if matrix_method='pathplanning'
    :return: Optimized list of viewpoint IDs
    """
    print(f"Starting viewpoint order optimization with {len(viewpoints)} viewpoints...")
    print(f"Matrix calculation method: {matrix_method}")
    print(f"TSP solver method: {tsp_method}")
    if matrix_method == 'pathplanning':
        print(f"Path planning method for matrix: {path_planning_method}")
    
    total_start_time = time.time()
    
    # Calculate TSP distance matrix
    distance_matrix = calculate_tsp_matrix(viewpoints, obstacles, matrix_method, path_planning_method)
    
    # Solve TSP
    print(f"Solving TSP using {tsp_method} method...")
    solve_start_time = time.time()
    
    if tsp_method == 'dynamic_programming':
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    elif tsp_method == 'simulated_annealing':
        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    elif tsp_method == 'local_search':
        permutation, distance = solve_tsp_local_search(distance_matrix)
    else:
        # Default to simulated annealing
        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    
    solve_end_time = time.time()
    solve_time = solve_end_time - solve_start_time
    
    # Get ordered viewpoint IDs
    viewpoint_ids = list(viewpoints.keys())
    optimized_ids = [viewpoint_ids[i] for i in permutation]
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"TSP solved in {solve_time:.2f} seconds.")
    print(f"Total optimization time: {total_time:.2f} seconds.")
    print(f"Optimized path length: {distance:.2f} units.")
    
    return optimized_ids, distance


def plan_path_between_viewpoints(start_pos, goal_pos, obstacles, method=DEFAULT_PATH_PLANNING):
    """
    Plan a path between two viewpoints avoiding obstacles.
    
    :param start_pos: 3D start position [x, y, z]
    :param goal_pos: 3D goal position [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param method: Path planning method to use
    :return: List of waypoints
    """
    if method == 'direct':
        # Check if direct path is collision-free
        if is_collision_free(start_pos, goal_pos, obstacles):
            return [start_pos, goal_pos]
        else:
            # Direct path not possible, use path planning
            return a_star_path_planning(start_pos, goal_pos, obstacles)
    elif method == 'a_star':
        return a_star_path_planning(start_pos, goal_pos, obstacles)
    else:
        # Default to A* path planning
        return a_star_path_planning(start_pos, goal_pos, obstacles)


def convert_waypoints_to_path_msg(waypoints, frame_id='earth'):
    """
    Convert a list of waypoints to a Path message for follow_path.
    
    :param waypoints: List of [x, y, z] positions
    :param frame_id: Reference frame ID for the path
    :return: Path message
    """
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Header
    import rclpy.time
    
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rclpy.time.Time().to_msg()
    
    for waypoint in waypoints:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = rclpy.time.Time().to_msg()
        pose.pose.position.x = float(waypoint[0])
        pose.pose.position.y = float(waypoint[1])
        pose.pose.position.z = float(waypoint[2])
        # Orientation will be handled by the yaw_mode in follow_path
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        
        path_msg.poses.append(pose)
    
    return path_msg


def drone_run(drone_interface, scenario, path_planning=DEFAULT_PATH_PLANNING, 
             tsp_method=DEFAULT_TSP_METHOD, matrix_method=DEFAULT_TSP_MATRIX_METHOD, 
             graph_type='full', logger=None) -> bool:
    """
    Run the mission for a single drone using path planning and TSP optimization.

    :param drone_interface: DroneInterface object
    :param scenario: Dictionary containing scenario information
    :param path_planning: Path planning method to use for actual navigation
    :param tsp_method: TSP solver method to use
    :param matrix_method: Method to calculate TSP matrix distances: 'euclidean' or 'pathplanning'
    :param graph_type: Type of graph representation to use
    :param logger: Optional logger for mission logging
    :return: Bool indicating if the mission was successful
    """
    print('Run mission with path planning and TSP optimization')
    if logger:
        logger.info(f'Starting mission with path planning: {path_planning}, ' 
                    f'TSP: {tsp_method}, Matrix method: {matrix_method}')
    
    # Get viewpoints and obstacles from scenario
    viewpoints = scenario.get("viewpoint_poses", {})
    obstacles = scenario.get("obstacles", {})
    
    if not viewpoints:
        print("No viewpoints found in scenario")
        return False
    
    # Start metrics collection
    start_time = time.time()
    total_distance = 0
    visited_markers = set()
    
    # Get current drone position as starting point
    current_pos = drone_interface.position #This is right， stick to it
    if current_pos is None:
        print("Unable to get current drone position")
        if logger:
            logger.error("Unable to get current drone position")
        return False
    
    # Optimize viewpoint order using TSP
    print(f"Optimizing viewpoint order using {tsp_method} TSP solver and {matrix_method} distance calculation")
    optimized_ids, estimated_distance = optimize_viewpoint_order(
        viewpoints, obstacles, tsp_method, matrix_method, path_planning)
    print(f"Optimized order: {optimized_ids}")
    print(f"Estimated total distance: {estimated_distance:.2f} meters")
    
    if logger:
        logger.info(f"Optimized viewpoint order: {optimized_ids}")
        logger.info(f"Estimated mission distance: {estimated_distance:.2f} meters")
    
    # Visit each viewpoint in optimized order
    for index, vp_id in enumerate(optimized_ids):
        vp = viewpoints[vp_id]
        goal_pos = [vp["x"], vp["y"], vp["z"]]
        vp_yaw = vp["w"]
        
        print(f"Going to viewpoint {vp_id} ({index+1}/{len(optimized_ids)})")
        if logger:
            logger.info(f"Moving to viewpoint {vp_id} at position {goal_pos}")
        
        # Plan path to the next viewpoint - always using the specified path planning method
        # This is where we use A* or other path planning methods regardless of how TSP was calculated
        path = plan_path_between_viewpoints(
            current_pos, goal_pos, obstacles, method=path_planning)
        
        if not path:
            print(f"Failed to plan path to viewpoint {vp_id}")
            if logger:
                logger.error(f"Failed to plan path to viewpoint {vp_id}")
            continue
        
        # Calculate path distance
        path_distance = 0
        for i in range(len(path) - 1):
            leg_distance = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            path_distance += leg_distance
        
        print(f"Path planned with {len(path)} waypoints, distance: {path_distance:.2f}m")
        if logger:
            logger.info(f"Path planned with {len(path)} waypoints, distance: {path_distance:.2f}m")
        
        # Convert waypoints to Path message for follow_path
        path_msg = convert_waypoints_to_path_msg(path)
        
        # Follow path to the waypoint (keep original yaw during path)
        print(f"Following path to viewpoint {vp_id}")
        from as2_msgs.msg import YawMode
        
        # First follow path with KEEP_YAW to reach the position
        success = drone_interface.follow_path(
            path=path_msg,
            speed=SPEED,
            frame_id='earth',
            yaw_mode=YawMode.KEEP_YAW,
            yaw_angle=float(vp_yaw),
            wait=True
        )
        
        if not success:
            print(f"Failed to follow path to viewpoint {vp_id}")
            if logger:
                logger.error(f"Failed to follow path to viewpoint {vp_id}")
            continue
        
        # Then rotate to the desired yaw at the final position
        print(f"Adjusting yaw at viewpoint {vp_id} to {vp_yaw}")
        success = drone_interface.go_to.go_to_point_with_yaw(goal_pos, angle=vp_yaw, speed=SPEED)
        
        if not success:
            print(f"Failed to adjust yaw at viewpoint {vp_id}")
            if logger:
                logger.warning(f"Failed to adjust yaw at viewpoint {vp_id}, continuing mission")
        
        # Update current position
        current_pos = goal_pos
        total_distance += path_distance
        
        # Detect ArUco markers at this viewpoint
        if isinstance(drone_interface, ArucoDetectorDrone):
            # Add a short delay to ensure the drone is stable and camera feed is updated
            print(f"Waiting to stabilize at viewpoint {vp_id} before scanning...")
            sleep(1.0)
            
            # Print current drone position and orientation for debugging
            current_pos = drone_interface.get_position()
            current_orientation = drone_interface.get_orientation()
            print(f"Current drone position: {current_pos}")
            print(f"Current drone orientation: {current_orientation}")
            
            print(f"Scanning for ArUco markers at viewpoint {vp_id} for {SCAN_DURATION} seconds...")
            if logger:
                logger.info(f"Scanning for ArUco markers at viewpoint {vp_id}")
            
            detected = drone_interface.scan_for_markers(duration=SCAN_DURATION)
            
            # Log results with more details
            print(f"Scan completed at viewpoint {vp_id}")
            if detected:
                visited_markers.update(detected)
                print(f"Detected markers: {detected}")
                print(f"Total markers detected so far: {visited_markers}")
                if logger:
                    logger.info(f"Detected markers at viewpoint {vp_id}: {detected}")
                    logger.info(f"Total markers detected so far: {len(visited_markers)}")
            else:
                print(f"No markers detected at viewpoint {vp_id}, check camera positioning and lighting")
                if logger:
                    logger.warning(f"No markers detected at viewpoint {vp_id}")
        
        print(f"Viewpoint {vp_id} visited successfully")
        sleep(SLEEP_TIME)
    
    # Calculate mission metrics
    mission_duration = time.time() - start_time
    average_speed = total_distance / mission_duration if mission_duration > 0 else 0
    
    print("\n=== Mission Summary ===")
    print(f"Total distance traveled: {total_distance:.2f} meters")
    print(f"Mission duration: {mission_duration:.2f} seconds")
    print(f"Average speed: {average_speed:.2f} m/s")
    print(f"Detected markers: {visited_markers}")
    print(f"Total markers detected: {len(visited_markers)}")
    
    if logger:
        logger.info("=== Mission Summary ===")
        logger.info(f"Total distance traveled: {total_distance:.2f} meters")
        logger.info(f"Mission duration: {mission_duration:.2f} seconds")
        logger.info(f"Average speed: {average_speed:.2f} m/s")
        logger.info(f"Detected markers: {visited_markers}")
        logger.info(f"Total markers detected: {len(visited_markers)}")
    
    return True


def drone_end(drone_interface: DroneInterface, logger=None) -> bool:
    """
    End the mission for a single drone.

    :param drone_interface: DroneInterface object
    :param logger: Optional logger for mission logging
    :return: Bool indicating if the land was successful
    """
    print('End mission')
    if logger:
        logger.info('Ending mission')

    # Land
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    if logger:
        logger.info(f'Land success: {success}')
    if not success:
        return success

    # Manual
    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')
    if logger:
        logger.info(f'Manual success: {success}')

    return success


def read_scenario(file_path):
    """
    Read the YAML scenario file.
    
    :param file_path: Path to the scenario file
    :return: Dictionary containing scenario information
    """
    try:
        with open(file_path, 'r') as file:
            scenario = yaml.safe_load(file)
        return scenario
    except Exception as e:
        print(f"Error reading scenario file: {str(e)}")
        return None


def visualize_scenario(scenario, path=None, save_path=None):
    """
    Visualize scenario with viewpoints, obstacles, and planned path.
    
    :param scenario: Dictionary containing scenario information
    :param path: Optional list of waypoints representing the planned path
    :param save_path: Path to save the visualization image
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot viewpoints
    viewpoints = scenario.get("viewpoint_poses", {})
    vp_x, vp_y, vp_z = [], [], []
    for vp_id, vp in viewpoints.items():
        vp_x.append(vp['x'])
        vp_y.append(vp['y'])
        vp_z.append(vp['z'])
        ax.text(vp['x'], vp['y'], vp['z'], f"VP{vp_id}", color='blue')
    
    ax.scatter(vp_x, vp_y, vp_z, c='blue', marker='o', s=100, label='Viewpoints')
    
    # Plot obstacles
    obstacles = scenario.get("obstacles", {})
    for obs_id, obs in obstacles.items():
        # Create cube vertices
        x, y, z = obs['x'], obs['y'], obs['z']
        dx, dy, dz = obs['w']/2, obs['d']/2, obs['h']/2
        
        # Plot cube
        for i, j, k in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                         (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
            ax.scatter(x + i*dx, y + j*dy, z + k*dz, c='red', marker='.', s=20)
        
        # Draw cube edges
        for i in [-1, 1]:
            for j in [-1, 1]:
                ax.plot([x + i*dx, x + i*dx], [y + j*dy, y + j*dy], [z - dz, z + dz], 'r-', alpha=0.5)
                ax.plot([x + i*dx, x + i*dx], [y - dy, y + dy], [z + j*dz, z + j*dz], 'r-', alpha=0.5)
                ax.plot([x - dx, x + dx], [y + i*dy, y + i*dy], [z + j*dz, z + j*dz], 'r-', alpha=0.5)
    
    # Plot drone start position
    if "drone_start_pose" in scenario:
        start = scenario["drone_start_pose"]
        ax.scatter([start['x']], [start['y']], [start['z']], c='green', marker='*', s=200, label='Drone Start')
    
    # Plot path if provided
    if path:
        path_x, path_y, path_z = [], [], []
        for point in path:
            path_x.append(point[0])
            path_y.append(point[1])
            path_z.append(point[2])
        
        ax.plot(path_x, path_y, path_z, 'g-', linewidth=2, label='Planned Path')
    
    # Set labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Mission Scenario Visualization')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = max([max(vp_x) - min(vp_x), max(vp_y) - min(vp_y), max(vp_z) - min(vp_z)])
    mid_x = (max(vp_x) + min(vp_x)) * 0.5
    mid_y = (max(vp_y) + min(vp_y)) * 0.5
    mid_z = (max(vp_z) + min(vp_z)) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Read scenario file
        scenario = read_scenario(args.scenario)
        if not scenario:
            print("Failed to read scenario file")
            return
        
        # Create logger if needed
        logger = None
        if args.log_dir:
            from mission_logger import MissionLogger
            os.makedirs(args.log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = os.path.join(args.log_dir, f"mission_log_{timestamp}.txt")
            logger = MissionLogger(log_file=log_file)
            logger.set_scenario_info(os.path.basename(args.scenario), args.path_planning, args.tsp_method)
        
        # Visualize scenario if requested
        if args.visualize:
            visualize_scenario(scenario)
        
        # Initialize ROS and drone interface
        rclpy.init(args=args.ros_args)
        
        if args.detect_markers:
            # Use the ArUco detector drone interface
            from drone_camera import ArucoDetectorDrone
            drone_interface = ArucoDetectorDrone(
                drone_id=args.namespace,
                verbose=args.verbose,
                use_sim_time=args.use_sim_time
            )
        else:
            # Use the standard drone interface
            from as2_python_api.drone_interface import DroneInterface
            drone_interface = DroneInterface(
                drone_id=args.namespace,
                verbose=args.verbose,
                use_sim_time=args.use_sim_time
            )
        
        # Run the mission
        drone_start(drone_interface, logger)
        
        mission_success = drone_run(
            drone_interface, 
            scenario, 
            path_planning=args.path_planning, 
            tsp_method=args.tsp_method,
            matrix_method=args.matrix_method,
            logger=logger
        )
        
        drone_end(drone_interface, logger)
        
        # Save mission report
        if logger:
            report_file = os.path.join(args.log_dir, f"mission_report_{timestamp}.json")
            logger.save_mission_report(report_file)
        
        rclpy.shutdown()
        
        return 0 if mission_success else 1
    
    except Exception as e:
        print(f"Error during mission: {str(e)}")
        traceback.print_exc()
        return 1

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UAV Mission Planning')
    
    parser.add_argument('scenario', help='Path to scenario YAML file')
    parser.add_argument('-n', '--namespace', default='drone0', help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', help='Use simulation time')
    parser.add_argument('-p', '--path_planning', default=DEFAULT_PATH_PLANNING, 
                        choices=['direct', 'a_star'], help='Path planning method to use')
    parser.add_argument('-t', '--tsp_method', default=DEFAULT_TSP_METHOD, 
                        choices=['dynamic_programming', 'simulated_annealing', 'local_search'], 
                        help='TSP solver method to use')
    parser.add_argument('-m', '--matrix_method', default=DEFAULT_TSP_MATRIX_METHOD, 
                        choices=['euclidean', 'pathplanning'], 
                        help='Method to calculate TSP distances: euclidean or pathplanning')
    parser.add_argument('--visualize', action='store_true', help='Visualize the scenario')
    parser.add_argument('--detect_markers', action='store_true', help='Enable ArUco marker detection')
    parser.add_argument('--log_dir', default=None, help='Directory to store mission logs')
    
    # Parse ROS args too
    args, ros_args = parser.parse_known_args()
    args.ros_args = ros_args
    
    return args


if __name__ == '__main__':
    main()
