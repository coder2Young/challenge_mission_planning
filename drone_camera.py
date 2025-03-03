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
                if self.current_image is not None:
                    # Make a copy of the image to avoid threading issues
                    image = self.current_image.copy()
                    print(f"Got image for processing, shape: {image.shape}")
                    break
            time.sleep(0.1)
        else:
            self.get_logger().warning("Timeout waiting for camera image")
            print("Timeout waiting for camera image - no image received within timeout period")
            return set()
        
        # Detect ArUco markers using older OpenCV API
        try:
            # Add debugging info about image
            print(f"Processing image for ArUco detection, image shape: {image.shape}, dtype: {image.dtype}")
            
            # Apply some preprocessing to help with detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding to help with marker detection
            # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
            
            print(f"ArUco detection complete: found {len(corners) if corners else 0} markers")
        except Exception as e:
            print(f"Error during ArUco detection: {str(e)}")
            return set()
        
        # Process results
        detected_ids = set()
        if ids is not None and len(ids) > 0:
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
            print(f"Successfully detected ArUco markers: {detected_ids}")
        else:
            self.get_logger().info("No ArUco markers detected")
            print("No ArUco markers detected in this frame")
            
        return detected_ids
    
    def scan_for_markers(self, duration=2.0, display_image=True):
        """
        Actively scan for markers for a specified duration.
        
        :param duration: How long to scan for markers (seconds)
        :param display_image: Whether to display detection images
        :return: Set of all marker IDs detected during scan
        """
        self.get_logger().info(f"Scanning for ArUco markers for {duration} seconds...")
        print(f"Starting {duration} second scan for ArUco markers...")
        
        all_detected = set()
        end_time = time.time() + duration
        scan_count = 0
        
        # Scan until duration expires
        while time.time() < end_time:
            # Detect markers in current frame
            scan_count += 1
            print(f"Scan iteration {scan_count}")
            detected = self.detect_aruco_markers(display_image=display_image)
            all_detected.update(detected)
            
            # Short pause between detections
            time.sleep(0.1)
            
        self.get_logger().info(f"Scan complete. Total markers detected: {len(all_detected)}")
        print(f"Scan complete. Performed {scan_count} detection iterations. Total markers detected: {all_detected}")
        return all_detected 