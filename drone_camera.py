"""
Drone interface with camera integration for ArUco marker detection.
"""

import time
import rclpy
import numpy as np
from as2_python_api.drone_interface import DroneInterface

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
import threading

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
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Detected ArUco marker IDs
        self.detected_markers = set()
        
        # Subscribe to camera feed
        self.create_subscription(
            Image, 
            f"{drone_id}/sensor_measurements/hd_camera/image_raw", 
            self.camera_callback, 
            qos_profile_sensor_data
        )
        
        print(f"Camera subscription initiated for {drone_id}/sensor_measurements/hd_camera/image_raw")

    def camera_callback(self, msg):
        """
        Process incoming camera images.
        
        :param msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV format
            with self.image_lock:
                self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {str(e)}")

    def detect_aruco_markers(self, display_image=False, timeout=3.0):
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
                if self.current_image is not None:
                    # Make a copy of the image to avoid threading issues
                    image = self.current_image.copy()
                    break
            time.sleep(0.1)
        else:
            self.get_logger().warning("Timeout waiting for camera image")
            return set()
        
        # Detect ArUco markers using older OpenCV API
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.aruco_params)
        
        # Process results
        detected_ids = set()
        if ids is not None:
            for marker_id in ids.flatten():
                detected_ids.add(int(marker_id))
                self.detected_markers.add(int(marker_id))
            
            # Display image with marker detections if requested
            if display_image and len(corners) > 0:
                # Draw detected markers
                image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
                
                # Display the image
                cv2.imshow("ArUco Marker Detection", image_with_markers)
                cv2.waitKey(1)  # Wait 1ms
        
        # Log detection results
        if detected_ids:
            self.get_logger().info(f"Detected ArUco markers: {detected_ids}")
        else:
            self.get_logger().info("No ArUco markers detected")
            
        return detected_ids
    
    def scan_for_markers(self, duration=2.0, display_image=True):
        """
        Actively scan for markers for a specified duration.
        
        :param duration: How long to scan for markers (seconds)
        :param display_image: Whether to display detection images
        :return: Set of all marker IDs detected during scan
        """
        self.get_logger().info(f"Scanning for ArUco markers for {duration} seconds...")
        
        all_detected = set()
        end_time = time.time() + duration
        
        # Scan until duration expires
        while time.time() < end_time:
            # Detect markers in current frame
            detected = self.detect_aruco_markers(display_image=display_image)
            all_detected.update(detected)
            
            # Short pause between detections
            time.sleep(0.1)
            
        self.get_logger().info(f"Scan complete. Total markers detected: {len(all_detected)}")
        return all_detected 