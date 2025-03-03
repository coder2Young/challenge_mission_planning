#!/usr/bin/env python3

"""
Utility functions for reading and processing scenario files.
"""

import os
import yaml
import numpy as np

def read_scenario(file_path):
    """
    Read a scenario file in YAML format.
    
    :param file_path: Path to the scenario file
    :return: Dictionary containing scenario information or None if error
    """
    try:
        with open(file_path, 'r') as file:
            scenario = yaml.safe_load(file)
        
        # Extract scenario name from file path if not provided
        if 'name' not in scenario:
            scenario_name = os.path.basename(file_path)
            if scenario_name.endswith('.yaml') or scenario_name.endswith('.yml'):
                scenario_name = scenario_name[:-5]  # Remove extension
            scenario['name'] = scenario_name
        
        return scenario
    except Exception as e:
        print(f"Error reading scenario file: {str(e)}")
        return None

def get_scenario_bounds(scenario):
    """
    Calculate the 3D bounds of a scenario.
    
    :param scenario: Dictionary containing scenario information
    :return: Dictionary with min and max bounds for x, y, z
    """
    # Initialize bounds with the drone start position
    if 'drone_start_pose' in scenario:
        start = scenario['drone_start_pose']
        bounds = {
            'x_min': start['x'], 'x_max': start['x'],
            'y_min': start['y'], 'y_max': start['y'],
            'z_min': start['z'], 'z_max': start['z']
        }
    else:
        bounds = {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0
        }
    
    # Update bounds with viewpoints
    viewpoints = scenario.get('viewpoint_poses', {})
    for vp_id, vp in viewpoints.items():
        bounds['x_min'] = min(bounds['x_min'], vp['x'])
        bounds['x_max'] = max(bounds['x_max'], vp['x'])
        bounds['y_min'] = min(bounds['y_min'], vp['y'])
        bounds['y_max'] = max(bounds['y_max'], vp['y'])
        bounds['z_min'] = min(bounds['z_min'], vp['z'])
        bounds['z_max'] = max(bounds['z_max'], vp['z'])
    
    # Update bounds with obstacles
    obstacles = scenario.get('obstacles', {})
    for obs_id, obs in obstacles.items():
        # Calculate obstacle extent
        half_width = obs['w'] / 2
        half_depth = obs['d'] / 2
        half_height = obs['h'] / 2
        
        bounds['x_min'] = min(bounds['x_min'], obs['x'] - half_width)
        bounds['x_max'] = max(bounds['x_max'], obs['x'] + half_width)
        bounds['y_min'] = min(bounds['y_min'], obs['y'] - half_depth)
        bounds['y_max'] = max(bounds['y_max'], obs['y'] + half_depth)
        bounds['z_min'] = min(bounds['z_min'], obs['z'] - half_height)
        bounds['z_max'] = max(bounds['z_max'], obs['z'] + half_height)
    
    # Add some padding
    padding = 1.0  # 1 meter padding
    bounds['x_min'] -= padding
    bounds['x_max'] += padding
    bounds['y_min'] -= padding
    bounds['y_max'] += padding
    bounds['z_min'] -= padding
    bounds['z_max'] += padding
    
    return bounds

def check_for_collisions(position, obstacles, margin=0.5):
    """
    Check if a position collides with any obstacle.
    
    :param position: 3D position [x, y, z]
    :param obstacles: Dictionary of obstacles with position and dimensions
    :param margin: Safety margin around obstacles in meters
    :return: True if collision, False otherwise
    """
    if not obstacles:
        return False
    
    # Convert to numpy array for easier manipulation
    position = np.array(position)
    
    # Check collision with each obstacle
    for obs_id, obs in obstacles.items():
        obs_pos = np.array([obs['x'], obs['y'], obs['z']])
        obs_size = np.array([obs['w'], obs['d'], obs['h']]) / 2 + margin
        
        # Check if position is inside obstacle's bounding box with margin
        if (abs(position[0] - obs_pos[0]) < obs_size[0] and
            abs(position[1] - obs_pos[1]) < obs_size[1] and
            abs(position[2] - obs_pos[2]) < obs_size[2]):
            return True
    
    return False

def validate_scenario(scenario):
    """
    Validate a scenario for potential issues.
    
    :param scenario: Dictionary containing scenario information
    :return: Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check required keys
    required_keys = ['drone_start_pose', 'viewpoint_poses']
    for key in required_keys:
        if key not in scenario:
            issues.append(f"Missing required key: {key}")
    
    if 'viewpoint_poses' in scenario and not scenario['viewpoint_poses']:
        issues.append("No viewpoints defined in scenario")
    
    # Check for collision between drone start position and obstacles
    if 'drone_start_pose' in scenario and 'obstacles' in scenario:
        start_pos = [
            scenario['drone_start_pose']['x'],
            scenario['drone_start_pose']['y'],
            scenario['drone_start_pose']['z']
        ]
        if check_for_collisions(start_pos, scenario['obstacles']):
            issues.append("Drone start position collides with an obstacle")
    
    # Check for collision between viewpoints and obstacles
    if 'viewpoint_poses' in scenario and 'obstacles' in scenario:
        for vp_id, vp in scenario['viewpoint_poses'].items():
            vp_pos = [vp['x'], vp['y'], vp['z']]
            if check_for_collisions(vp_pos, scenario['obstacles']):
                issues.append(f"Viewpoint {vp_id} collides with an obstacle")
    
    return len(issues) == 0, issues

def extract_viewpoint_coordinates(scenario):
    """
    Extract coordinates of all viewpoints in a scenario.
    
    :param scenario: Dictionary containing scenario information
    :return: List of dictionaries with viewpoint coordinates
    """
    viewpoints = []
    
    if 'viewpoint_poses' in scenario:
        for vp_id, vp in scenario['viewpoint_poses'].items():
            viewpoints.append({
                'id': vp_id,
                'x': vp['x'],
                'y': vp['y'],
                'z': vp['z'],
                'yaw': vp.get('w', 0.0)
            })
    
    return viewpoints 