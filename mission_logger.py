#!/usr/bin/env python3

"""
Mission logger for UAV mission planning.

This module provides a logging facility for recording mission metrics,
detected markers, and path information during mission execution.
"""

import os
import datetime
import logging
import json
import numpy as np
import time

class MissionLogger:
    """Logger for mission metrics and data collection."""
    
    def __init__(self, log_file=None, log_level='INFO'):
        """
        Initialize the mission logger.
        
        :param log_file: Path to log file (if None, generate a timestamped file in logs dir)
        :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Generate log file name if not provided
        if log_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"mission_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger('mission_logger')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Mission metrics
        self.mission_start_time = time.time()
        self.viewpoints_visited = []
        self.markers_detected = set()
        self.total_distance = 0.0
        self.path_waypoints = []
        
        # Metadata
        self.scenario_name = None
        self.path_planning_method = None
        self.tsp_method = None
        
        self.info(f"Mission logger initialized. Logging to {log_file}")
        
    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
        
    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
        
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
        
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
        
    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)
    
    def set_scenario_info(self, scenario_name, path_planning_method, tsp_method):
        """
        Set scenario and planning information.
        
        :param scenario_name: Name of the scenario
        :param path_planning_method: Path planning method used
        :param tsp_method: TSP method used
        """
        self.scenario_name = scenario_name
        self.path_planning_method = path_planning_method
        self.tsp_method = tsp_method
        self.info(f"Scenario: {scenario_name}, Path planning: {path_planning_method}, TSP: {tsp_method}")
    
    def log_viewpoint_visit(self, viewpoint_id, position):
        """
        Log a viewpoint visit.
        
        :param viewpoint_id: ID of the viewpoint
        :param position: 3D position of the viewpoint
        """
        self.viewpoints_visited.append({
            'id': viewpoint_id,
            'position': position,
            'timestamp': time.time() - self.mission_start_time
        })
        self.info(f"Visited viewpoint {viewpoint_id} at position {position}")
    
    def log_marker_detection(self, marker_ids, viewpoint_id=None):
        """
        Log detected ArUco markers.
        
        :param marker_ids: Set or list of detected marker IDs
        :param viewpoint_id: Optional viewpoint ID where markers were detected
        """
        if not marker_ids:
            return
            
        self.markers_detected.update(marker_ids)
        if viewpoint_id:
            self.info(f"Detected markers {marker_ids} at viewpoint {viewpoint_id}")
        else:
            self.info(f"Detected markers: {marker_ids}")
    
    def log_path_segment(self, start_pos, end_pos, distance):
        """
        Log a path segment.
        
        :param start_pos: Start position of path segment
        :param end_pos: End position of path segment
        :param distance: Length of path segment in meters
        """
        self.path_waypoints.append({
            'start': start_pos,
            'end': end_pos,
            'distance': distance,
            'timestamp': time.time() - self.mission_start_time
        })
        self.total_distance += distance
        self.debug(f"Path segment: {start_pos} to {end_pos}, distance: {distance:.2f}m")
    
    def save_mission_report(self, file_path=None):
        """
        Save a mission report with all collected metrics.
        
        :param file_path: Path to save report (if None, generate a timestamped file)
        :return: Path to the saved report
        """
        # Generate report file name if not provided
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join('logs', f"mission_report_{timestamp}.json")
        
        # Calculate mission metrics
        mission_duration = time.time() - self.mission_start_time
        average_speed = self.total_distance / mission_duration if mission_duration > 0 else 0
        
        # Prepare report data
        report = {
            'scenario_name': self.scenario_name,
            'path_planning_method': self.path_planning_method,
            'tsp_method': self.tsp_method,
            'mission_duration': mission_duration,
            'total_distance': self.total_distance,
            'average_speed': average_speed,
            'viewpoints_visited': len(self.viewpoints_visited),
            'viewpoints_details': self.viewpoints_visited,
            'markers_detected': list(self.markers_detected),
            'total_markers_detected': len(self.markers_detected),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save report to file
        try:
            with open(file_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump(report, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            self.info(f"Mission report saved to {file_path}")
            return file_path
        except Exception as e:
            self.error(f"Error saving mission report: {str(e)}")
            return None
    
    def get_mission_summary(self):
        """
        Get a formatted summary of the mission.
        
        :return: String containing mission summary
        """
        mission_duration = time.time() - self.mission_start_time
        average_speed = self.total_distance / mission_duration if mission_duration > 0 else 0
        
        summary = [
            "=== Mission Summary ===",
            f"Scenario: {self.scenario_name}",
            f"Path planning method: {self.path_planning_method}",
            f"TSP method: {self.tsp_method}",
            f"Total distance traveled: {self.total_distance:.2f} meters",
            f"Mission duration: {mission_duration:.2f} seconds",
            f"Average speed: {average_speed:.2f} m/s",
            f"Viewpoints visited: {len(self.viewpoints_visited)}",
            f"Detected markers: {sorted(list(self.markers_detected))}",
            f"Total markers detected: {len(self.markers_detected)}"
        ]
        
        return "\n".join(summary) 