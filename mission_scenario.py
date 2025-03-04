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
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search, solve_tsp_lin_kernighan
from as2_msgs.msg import YawMode
from as2_python_api.drone_interface import DroneInterface
#from drone_camera import ArucoDetectorDrone
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading

# Mission parameters
TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0  # Max speed in m/s
SLEEP_TIME = 0.5  # Sleep time between behaviors in seconds
SPEED = 1.0  # Max speed in m/s
LAND_SPEED = 0.5  # Max speed in m/s
OBSTACLE_SAFETY_MARGIN = 0.5  # Safety margin around obstacles in meters

# Path planning methods
PATH_PLANNING_METHODS = ['direct', 'a_star', 'dijkstra', 'rrt']
DEFAULT_PATH_PLANNING = 'rrt'

# TSP solver methods
TSP_METHODS = ['dynamic_programming', 'simulated_annealing', 'local_search', 'brute_force', 'lin_kernighan']
DEFAULT_TSP_METHOD = 'simulated_annealing'

# Default parameters
DEFAULT_TSP_MATRIX_METHOD = 'pathplanning'  # Default TSP matrix calculation method: 'euclidean' or 'pathplanning'
DEFAULT_ASTAR_IMPLEMENTATION = 'optimized'  # Default A* implementation: 'optimized' or 'networkx'


"""
Drone interface with camera integration for ArUco marker detection.
"""
class ArucoDetectorDrone(DroneInterface):
    """
    DroneInterface extension with ArUco marker detection capabilities.
    
    This class extends the standard DroneInterface to add camera integration
    and ArUco marker detection functionality.
    """

    def __init__(self, drone_id: str = 'drone0', verbose: bool = False,
                 use_sim_time: bool = False, spin_rate: float = 20.0):
        """
        Initialize the drone interface with camera capabilities.
        
        :param drone_id: drone namespace, defaults to "drone0"
        :param verbose: output mode, defaults to False
        :param use_sim_time: use simulation time, defaults to False
        :param spin_rate: spin rate (Hz), defaults to 20.0
        """
        super().__init__(drone_id=drone_id, verbose=verbose, 
                         use_sim_time=use_sim_time, spin_rate=spin_rate)
        
        # Initialize CV Bridge for ROS image to OpenCV conversion
        self.cv_bridge = CvBridge()
        
        # Most recent camera image
        self.current_image = None
        self.image_lock = threading.Lock()
        
        # ArUco detector parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        # Detected ArUco marker IDs
        self.detected_markers = set()
        
        # Subscribe to camera feed
        self.create_subscription(
            Image, 
            "sensor_measurements/hd_camera/image_raw", 
            self.camera_callback, 
            qos_profile_sensor_data
        )
        
        print(f"Camera subscription initiated for sensor_measurements/hd_camera/image_raw")

    def camera_callback(self, msg):
        """
        Process incoming camera images.
        
        :param msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV format
            with self.image_lock:
                self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                #print(f"Received camera image, shape: {self.current_image.shape if self.current_image is not None else None}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {str(e)}")
            print(f"Error processing camera image: {str(e)}")

    def detect_aruco_markers(self, timeout=3.0):
        """
        Detect ArUco markers in the current camera image.
        
        :param display_image: Whether to display the image with detected markers
        :param timeout: Maximum time to wait for a valid image (seconds)
        :return: Set of detected marker IDs
        """
        start_time = time.time()
        
        # Wait for a valid image, but no longer than timeout
        while time.time() - start_time < timeout:
            with self.image_lock:
                if self.current_image is None:
                    continue;

                # Make a copy of the image to avoid threading issues
                image = self.current_image.copy()
                #print(f"Got image for processing, shape: {image.shape}")
                # Apply some preprocessing to help with detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params)
                
                if ids is not None and len(ids) > 0:
                    #print(f"ArUco detection complete: found {len(corners) if corners else 0} markers")
                    return True, ids.flatten()
            
        return False, None

def drone_start(drone_interface: DroneInterface) -> bool:
    """
    Take off the drone.

    :param drone_interface: DroneInterface object
    :return: Bool indicating if the take off was successful
    """
    print('Start mission')

    # Arm
    print('Arm')
    success = drone_interface.arm()
    print(f'Arm success: {success}')

    # Offboard
    print('Offboard')
    success = drone_interface.offboard()
    print(f'Offboard success: {success}')

    # Take Off
    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'Take Off success: {success}')

    # Init position [0, 0, TAKE_OFF_HEIGHT]
    # Back to the initial position if the drone is not there
    if drone_interface.position != [0, 0, TAKE_OFF_HEIGHT]:
        print('Back to initial position')
        drone_interface.go_to(x=0, y=0, z=TAKE_OFF_HEIGHT, speed=SPEED)

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


def a_star_path_planning_networkx(start, goal, obstacles, margin=OBSTACLE_SAFETY_MARGIN):
    """
    Implement A* algorithm for path planning using NetworkX.
    
    :param start: 3D start point [x, y, z]
    :param goal: 3D end point [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :return: List of waypoints including start and goal
    """
    # Check if direct path is possible
    if is_collision_free(start, goal, obstacles, margin):
        return [start, goal]
    
    print("Starting NetworkX based A* path planning...")
    start_time = time.time()
    
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
        end_time = time.time()
        print(f"NetworkX A* path found with {len(path)} waypoints")
        print(f"NetworkX A* search completed in {end_time - start_time:.3f} seconds")
        return path
    except nx.NetworkXNoPath:
        # If no path found, try with direct connection
        end_time = time.time()
        print(f"NetworkX A* could not find a path, returning direct connection (in {end_time - start_time:.3f} seconds)")
        return [start, goal]


def a_star_path_planning(start, goal, obstacles, margin=OBSTACLE_SAFETY_MARGIN, implementation=DEFAULT_ASTAR_IMPLEMENTATION):
    """
    Implement A* algorithm for path planning between viewpoints.
    
    :param start: 3D start point [x, y, z]
    :param goal: 3D end point [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :param implementation: Which A* implementation to use ('optimized' or 'networkx')
    :return: List of waypoints including start and goal
    """
    # Redirect to the appropriate implementation
    if implementation == 'networkx':
        return a_star_path_planning_networkx(start, goal, obstacles, margin)
    
    # Default to optimized implementation
    # Check if direct path is possible
    if is_collision_free(start, goal, obstacles, margin):
        return [start, goal]
    
    print("Starting optimized A* path planning...")
    start_time = time.time()
    
    # Convert to tuples for hashability
    start = tuple(start)
    goal = tuple(goal)
    
    # Grid resolution and boundaries
    resolution = 1.0  # meters
    
    # Determine grid bounds
    bounds_min = [-15, -15, 0]
    bounds_max = [15, 15, 10]
    
    # Directions for 3D movement - 6 primary directions (up, down, north, south, east, west)
    # Can be extended to include diagonal movements if needed
    directions = [
        (resolution, 0, 0), (-resolution, 0, 0),  # East, West
        (0, resolution, 0), (0, -resolution, 0),  # North, South
        (0, 0, resolution), (0, 0, -resolution)   # Up, Down
    ]
    
    # Add diagonal movements in the horizontal plane
    diagonals = [
        (resolution, resolution, 0), (resolution, -resolution, 0),
        (-resolution, resolution, 0), (-resolution, -resolution, 0)
    ]
    directions.extend(diagonals)
    
    # A* data structures
    from heapq import heappush, heappop
    
    # Open set - priority queue of nodes to be evaluated
    open_set = []
    
    # Closed set - set of nodes already evaluated
    closed_set = set()
    
    # g_score - cost from start to current node
    g_score = {start: 0}
    
    # f_score - estimated total cost from start to goal through current node
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}
    
    # came_from - for path reconstruction
    came_from = {}
    
    # Add start node to open set
    heappush(open_set, (f_score[start], start))
    
    # For profiling
    nodes_evaluated = 0
    
    while open_set:
        # Get node with lowest f_score
        _, current = heappop(open_set)
        nodes_evaluated += 1
        
        # Check if we've reached the goal
        if np.linalg.norm(np.array(current) - np.array(goal)) < resolution:
            # Reconstruct path
            path = [goal]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            
            end_time = time.time()
            print(f"Optimized A* path found with {len(path)} waypoints")
            print(f"Optimized A* search evaluated {nodes_evaluated} nodes in {end_time - start_time:.3f} seconds")
            
            # Convert tuples back to lists
            return [list(p) for p in path]
        
        # Mark current node as processed
        closed_set.add(current)
        
        # Check all neighbors
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Skip if outside bounds
            if (neighbor[0] < bounds_min[0] or neighbor[0] > bounds_max[0] or
                neighbor[1] < bounds_min[1] or neighbor[1] > bounds_max[1] or
                neighbor[2] < bounds_min[2] or neighbor[2] > bounds_max[2]):
                continue
            
            # Skip if in closed set
            if neighbor in closed_set:
                continue
            
            # Check if neighbor is in an obstacle
            in_obstacle = False
            for obs_id, obs in obstacles.items():
                obs_pos = np.array([obs['x'], obs['y'], obs['z']])
                obs_size = np.array([obs['w'], obs['d'], obs['h']]) / 2 + margin
                
                if (abs(neighbor[0] - obs_pos[0]) < obs_size[0] and
                    abs(neighbor[1] - obs_pos[1]) < obs_size[1] and
                    abs(neighbor[2] - obs_pos[2]) < obs_size[2]):
                    in_obstacle = True
                    break
            
            if in_obstacle:
                continue
            
            # Check if direct path to neighbor is collision-free
            if not is_collision_free(current, neighbor, obstacles, margin):
                continue
            
            # Calculate tentative g_score
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor) - np.array(current))
            
            # Skip if this path to neighbor is worse
            if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                continue
            
            # Record this path as the best so far
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + np.linalg.norm(np.array(neighbor) - np.array(goal))
            
            # Add to open set if not already there
            if neighbor not in [i[1] for i in open_set]:
                heappush(open_set, (f_score[neighbor], neighbor))
    
    # If no path found
    end_time = time.time()
    print(f"Optimized A* search failed to find a path. Returning direct path. (in {end_time - start_time:.3f} seconds)")
    return [list(start), list(goal)]


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
    print("\n=== Viewpoint Order (TSP) Optimization ===")
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
    elif tsp_method == 'brute_force':
        permutation, distance = solve_tsp_brute_force(distance_matrix)
    elif tsp_method == 'lin_kernighan':
        permutation, distance = solve_tsp_lin_kernighan(distance_matrix)
    else:
        # Default to simulated annealing
        print(f"Unknown TSP method: {tsp_method}, defaulting to simulated annealing")
        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    
    solve_end_time = time.time()
    solve_time = solve_end_time - solve_start_time
    
    # Get ordered viewpoint IDs
    viewpoint_ids = list(viewpoints.keys())
    optimized_ids = [viewpoint_ids[i] for i in permutation]
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\nTSP solved in {solve_time:.2f} seconds.")
    print(f"Total optimization time: {total_time:.2f} seconds.")
    print(f"Optimized path length: {distance:.2f} meters.")
    
    return optimized_ids, distance


def dijkstra_path_planning(start, goal, obstacles, margin=OBSTACLE_SAFETY_MARGIN):
    """
    Implement Dijkstra's algorithm for path planning between viewpoints.
    
    :param start: 3D start point [x, y, z]
    :param goal: 3D end point [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :return: List of waypoints including start and goal
    """
    # Check if direct path is possible
    if is_collision_free(start, goal, obstacles, margin):
        return [start, goal]
    
    print("Starting Dijkstra path planning...")
    start_time = time.time()
    
    # Convert to tuples for hashability
    start = tuple(start)
    goal = tuple(goal)
    
    # Grid resolution and boundaries
    resolution = 1.0  # meters
    
    # Determine grid bounds
    bounds_min = [-15, -15, 0]
    bounds_max = [15, 15, 10]
    
    # Directions for 3D movement - 6 primary directions + diagonals
    directions = [
        (resolution, 0, 0), (-resolution, 0, 0),  # East, West
        (0, resolution, 0), (0, -resolution, 0),  # North, South
        (0, 0, resolution), (0, 0, -resolution)   # Up, Down
    ]
    
    # Add diagonal movements in the horizontal plane
    diagonals = [
        (resolution, resolution, 0), (resolution, -resolution, 0),
        (-resolution, resolution, 0), (-resolution, -resolution, 0)
    ]
    directions.extend(diagonals)
    
    # Dijkstra data structures
    from heapq import heappush, heappop
    
    # Priority queue for nodes to explore
    queue = []
    
    # Set of visited nodes
    visited = set()
    
    # Distance from start to each node
    distance = {start: 0}
    
    # For path reconstruction
    previous = {}
    
    # Start with the start node
    heappush(queue, (0, start))
    
    # For profiling
    nodes_evaluated = 0
    
    # Dijkstra's algorithm
    while queue:
        # Get node with smallest distance
        dist, current = heappop(queue)
        nodes_evaluated += 1
        
        # If we reached the goal or very close to it
        if np.linalg.norm(np.array(current) - np.array(goal)) < resolution:
            path = [goal]
            while current in previous:
                path.append(current)
                current = previous[current]
            path.reverse()
            
            end_time = time.time()
            print(f"Dijkstra path found with {len(path)} waypoints")
            print(f"Dijkstra search evaluated {nodes_evaluated} nodes in {end_time - start_time:.3f} seconds")
            
            # Convert tuples back to lists
            return [list(p) for p in path]
        
        # Skip if already visited
        if current in visited:
            continue
        
        # Mark as visited
        visited.add(current)
        
        # Explore neighbors
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Skip if outside bounds
            if (neighbor[0] < bounds_min[0] or neighbor[0] > bounds_max[0] or
                neighbor[1] < bounds_min[1] or neighbor[1] > bounds_max[1] or
                neighbor[2] < bounds_min[2] or neighbor[2] > bounds_max[2]):
                continue
            
            # Skip if already visited
            if neighbor in visited:
                continue
            
            # Check if neighbor is in an obstacle
            in_obstacle = False
            for obs_id, obs in obstacles.items():
                obs_pos = np.array([obs['x'], obs['y'], obs['z']])
                obs_size = np.array([obs['w'], obs['d'], obs['h']]) / 2 + margin
                
                if (abs(neighbor[0] - obs_pos[0]) < obs_size[0] and
                    abs(neighbor[1] - obs_pos[1]) < obs_size[1] and
                    abs(neighbor[2] - obs_pos[2]) < obs_size[2]):
                    in_obstacle = True
                    break
            
            if in_obstacle:
                continue
            
            # Check if direct path to neighbor is collision-free
            if not is_collision_free(current, neighbor, obstacles, margin):
                continue
            
            # Calculate new distance
            new_distance = distance[current] + np.linalg.norm(np.array(neighbor) - np.array(current))
            
            # Update if this path is better
            if neighbor not in distance or new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                previous[neighbor] = current
                heappush(queue, (new_distance, neighbor))
    
    # If no path found
    end_time = time.time()
    print(f"Dijkstra search failed to find a path. Returning direct path. (in {end_time - start_time:.3f} seconds)")
    return [list(start), list(goal)]


def rrt_path_planning(start, goal, obstacles, margin=OBSTACLE_SAFETY_MARGIN, max_iterations=5000, step_size=1.0):
    """
    Implement RRT (Rapidly-exploring Random Tree) for path planning.
    
    :param start: 3D start point [x, y, z]
    :param goal: 3D end point [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :param max_iterations: Maximum number of iterations for RRT
    :param step_size: Step size for extending the tree
    :return: List of waypoints including start and goal
    """
    # Check if direct path is possible
    if is_collision_free(start, goal, obstacles, margin):
        return [start, goal]
    
    print("Starting RRT path planning...")
    start_time = time.time()
    
    # Convert to numpy arrays
    start_np = np.array(start)
    goal_np = np.array(goal)
    
    # Bounds for random sampling
    bounds_min = np.array([-15, -15, 0])
    bounds_max = np.array([15, 15, 10])
    
    # Goal bias - probability of sampling the goal
    goal_bias = 0.1
    
    # Tree structure: vertices and edges
    vertices = [start_np]
    edges = {}  # parent -> child mapping for path reconstruction
    
    # For profiling
    iterations = 0
    
    # Main RRT loop
    for i in range(max_iterations):
        iterations += 1
        
        # Sample random point or goal with some probability
        if np.random.random() < goal_bias:
            random_point = goal_np
        else:
            random_point = np.array([
                np.random.uniform(bounds_min[0], bounds_max[0]),
                np.random.uniform(bounds_min[1], bounds_max[1]),
                np.random.uniform(bounds_min[2], bounds_max[2])
            ])
        
        # Find nearest vertex in the tree
        nearest_idx = np.argmin([np.linalg.norm(v - random_point) for v in vertices])
        nearest = vertices[nearest_idx]
        
        # Steer towards random point with limited step size
        direction = random_point - nearest
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance  # Normalize
            
            # Limit step size
            new_point = nearest + direction * min(step_size, distance)
            
            # Check if the new point is collision-free
            if not is_collision_free(nearest, new_point, obstacles, margin):
                continue
            
            # Add new point to the tree
            vertices.append(new_point)
            edges[len(vertices) - 1] = nearest_idx
            
            # Check if we can connect to goal
            if np.linalg.norm(new_point - goal_np) < step_size and is_collision_free(new_point, goal_np, obstacles, margin):
                # Path found! Reconstruct the path
                path = [goal_np]
                current_idx = len(vertices) - 1
                
                while current_idx != 0:  # Until we reach the start
                    path.append(vertices[current_idx])
                    current_idx = edges[current_idx]
                
                path.append(start_np)
                path.reverse()
                
                # Path smoothing - optional but recommended for RRT
                smoothed_path = path_smoothing(path, obstacles, margin)
                
                end_time = time.time()
                print(f"RRT path found in {iterations} iterations")
                print(f"RRT path has {len(smoothed_path)} waypoints")
                print(f"RRT search completed in {end_time - start_time:.3f} seconds")
                
                # Convert numpy arrays to lists
                return [p.tolist() for p in smoothed_path]
    
    # If no path found after max iterations
    end_time = time.time()
    print(f"RRT failed to find a path after {max_iterations} iterations. Returning direct path. (in {end_time - start_time:.3f} seconds)")
    return [start, goal]


def path_smoothing(path, obstacles, margin, max_iterations=50):
    """
    Smooth a path by removing unnecessary waypoints while keeping it collision-free.
    
    :param path: List of waypoints
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles
    :param max_iterations: Maximum number of smoothing iterations
    :return: Smoothed path
    """
    if len(path) <= 2:
        return path
    
    smoothed_path = path.copy()
    
    # Iteratively try to remove waypoints
    for _ in range(max_iterations):
        # Start with a random index (not the start or goal)
        if len(smoothed_path) <= 2:
            break
            
        i = np.random.randint(1, len(smoothed_path) - 1)
        
        # Try to remove the waypoint by checking if direct path is collision-free
        if is_collision_free(smoothed_path[i-1], smoothed_path[i+1], obstacles, margin):
            smoothed_path.pop(i)
    
    return smoothed_path


def plan_path_between_viewpoints(start_pos, goal_pos, obstacles, method=DEFAULT_PATH_PLANNING, astar_implementation=DEFAULT_ASTAR_IMPLEMENTATION):
    """
    Plan a path between two viewpoints avoiding obstacles.
    
    :param start_pos: 3D start position [x, y, z]
    :param goal_pos: 3D goal position [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param method: Path planning method to use
    :param astar_implementation: Which A* implementation to use ('optimized' or 'networkx')
    :return: List of waypoints
    """
    print("\n==== Path Planning ====")
    if method == 'direct':
        # Check if direct path is collision-free
        if is_collision_free(start_pos, goal_pos, obstacles):
            return [start_pos, goal_pos]
        else:
            # Direct path not possible, use path planning
            return a_star_path_planning(start_pos, goal_pos, obstacles, implementation=astar_implementation)
    elif method == 'a_star':
        return a_star_path_planning(start_pos, goal_pos, obstacles, implementation=astar_implementation)
    elif method == 'dijkstra':
        return dijkstra_path_planning(start_pos, goal_pos, obstacles)
    elif method == 'rrt':
        return rrt_path_planning(start_pos, goal_pos, obstacles)
    else:
        # Default to A* path planning
        return a_star_path_planning(start_pos, goal_pos, obstacles, implementation=astar_implementation)


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
             astar_implementation=DEFAULT_ASTAR_IMPLEMENTATION,
             graph_type='full') -> bool:
    """
    Run the mission for a single drone using path planning and TSP optimization.

    :param drone_interface: DroneInterface object
    :param scenario: Dictionary containing scenario information
    :param path_planning: Path planning method to use for actual navigation
    :param tsp_method: TSP solver method to use
    :param matrix_method: Method to calculate TSP matrix distances: 'euclidean' or 'pathplanning'
    :param astar_implementation: Which A* implementation to use ('optimized' or 'networkx')
    :param graph_type: Type of graph representation to use
    :return: Bool indicating if the mission was successful
    """
    print('=== Run mission with path planning and TSP optimization ===')
    
    # Get viewpoints and obstacles from scenario
    viewpoints = scenario.get("viewpoint_poses", {})
    obstacles = scenario.get("obstacles", {})
    
    # Start metrics collection
    start_time = time.time()
    total_distance = 0
    visited_markers = []
    
    # Get current drone position as starting point
    current_pos = drone_interface.position
    if current_pos is None:
        print("Unable to get current drone position")
        return False
    
    # Optimize viewpoint order using TSP
    print(f"Optimizing viewpoint order using {tsp_method} TSP solver and {matrix_method} distance calculation")
    optimized_ids, estimated_distance = optimize_viewpoint_order(
        viewpoints, obstacles, tsp_method, matrix_method, path_planning)
    print(f"Optimized order: {optimized_ids}")
    #print(f"Estimated total distance: {estimated_distance:.2f} meters")
    
    # Visit each viewpoint in optimized order
    for index, vp_id in enumerate(optimized_ids):
        vp = viewpoints[vp_id]
        goal_pos = [vp["x"], vp["y"], vp["z"]]
        vp_yaw = vp["w"]
        
        print(f"\n==== Viewpoint {vp_id} ====")
        #print(f"Going to viewpoint {vp_id} ({index+1}/{len(optimized_ids)})")
        
        # Plan path to the next viewpoint - always using the specified path planning method
        # This is where we use A* or other path planning methods regardless of how TSP was calculated
        path = plan_path_between_viewpoints(
            current_pos, goal_pos, obstacles, method=path_planning, astar_implementation=astar_implementation)
        
        if not path:
            print(f"Failed to plan path to viewpoint {vp_id}")
            continue
        
        # Calculate path distance
        path_distance = 0
        for i in range(len(path) - 1):
            leg_distance = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            path_distance += leg_distance
        
        print(f"Path planned with {len(path)} waypoints, distance: {path_distance:.2f}m")
        
        # Convert waypoints to Path message for follow_path
        path_msg = convert_waypoints_to_path_msg(path)
        
        # Follow path to the waypoint (keep original yaw during path)
        print(f"Following path to viewpoint {vp_id}")
        
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
            continue
        
        # Then rotate to the desired yaw at the final position
        #print(f"Adjusting yaw at viewpoint {vp_id} to {vp_yaw}")
        success = drone_interface.go_to.go_to_point_with_yaw(goal_pos, angle=vp_yaw, speed=SPEED)
        
        # Update current position
        current_pos = goal_pos
        total_distance += path_distance
        

        # Add a short delay to ensure the drone is stable and camera feed is updated
        #print(f"Waiting to stabilize at viewpoint {vp_id} before scanning...")
        #print(f"Scanning for ArUco markers at viewpoint {vp_id}...")
        
        detected, detected_ids = drone_interface.detect_aruco_markers()
        
        # Log results with more details
        #print(f"Scan completed at viewpoint {vp_id}")
        print("\n==== Aruco Marker Detection Results ====")
        if detected:
            print(f"Detected markers: {detected_ids} at viewpoint {vp_id}")
            visited_markers.append(detected_ids)
            print(f"Total markers detected so far: {len(visited_markers)}")
        else:
            print(f"No markers detected at viewpoint {vp_id}")

    # Calculate mission metrics
    mission_duration = time.time() - start_time
    average_speed = total_distance / mission_duration if mission_duration > 0 else 0
    
    print("\n=== Mission Summary ===")
    print(f"Total distance traveled: {total_distance:.2f} meters")
    print(f"Mission duration: {mission_duration:.2f} seconds")
    print(f"Average speed: {average_speed:.2f} m/s")
    #print(f"Detected markers: {visited_markers}")
    print(f"Total markers detected: {len(visited_markers)}")

    if len(visited_markers) == len(viewpoints):
        print("Mission successful")
        return True
    else:
        print("Mission failed")
        return False


def drone_end(drone_interface: DroneInterface) -> bool:
    """
    End the mission for a single drone.

    :param drone_interface: DroneInterface object
    :return: Bool indicating if the land was successful
    """
    print('End mission')

    # Land
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    if not success:
        return success

    # Manual
    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')

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
    
    # Read scenario file
    scenario = read_scenario(args.scenario)
    if not scenario:
        print("Failed to read scenario file")
        return
    
    # Initialize ROS and drone interface
    rclpy.init(args=args.ros_args)
    

    # Use the ArUco detector drone interface
    from drone_camera import ArucoDetectorDrone
    drone_interface = ArucoDetectorDrone(
        drone_id=args.namespace,
        verbose=args.verbose,
        use_sim_time=args.use_sim_time
    )

    
    # Run the mission
    drone_start(drone_interface)
    
    mission_success = drone_run(
        drone_interface, 
        scenario, 
        path_planning=args.path_planning, 
        tsp_method=args.tsp_method,
        matrix_method=args.matrix_method,
        astar_implementation=args.astar_implementation,
    )
    
    drone_end(drone_interface)
    
    rclpy.shutdown()
    
    return 0 if mission_success else 1
    

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UAV Mission Planning')
    
    parser.add_argument('scenario', help='Path to scenario YAML file')
    parser.add_argument('-n', '--namespace', default='drone0', help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', help='Use simulation time')
    parser.add_argument('-p', '--path_planning', default=DEFAULT_PATH_PLANNING, 
                        choices=PATH_PLANNING_METHODS, 
                        help='Path planning method to use')
    parser.add_argument('-t', '--tsp_method', default=DEFAULT_TSP_METHOD, 
                        choices=TSP_METHODS, 
                        help='TSP solver method to use')
    parser.add_argument('-m', '--matrix_method', default=DEFAULT_TSP_MATRIX_METHOD, 
                        choices=['euclidean', 'pathplanning'], 
                        help='Method to calculate TSP distances: euclidean or pathplanning')
    parser.add_argument('-a', '--astar_implementation', default=DEFAULT_ASTAR_IMPLEMENTATION,
                        choices=['optimized', 'networkx'],
                        help='A* implementation to use: optimized (default) or networkx')
    
    # Parse ROS args too
    args, ros_args = parser.parse_known_args()
    args.ros_args = ros_args
    
    return args


if __name__ == '__main__':
    main()
