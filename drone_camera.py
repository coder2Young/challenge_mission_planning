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
                    print(f"ArUco detection complete: found {len(corners) if corners else 0} markers")
                    return True, ids.flatten()
            
        return False, None