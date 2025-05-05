import cv2
import numpy as np
import os
import json

def find_chessboard_corners(image, pattern_size=(7, 7)):
    """
    Find chessboard corners in an image.
    
    Args:
        image (numpy.ndarray): Input image
        pattern_size (tuple): Pattern size as (columns, rows) of internal corners
        
    Returns:
        tuple: (success, corners) where corners is a numpy array of corner coordinates
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corners to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners
    else:
        return False, None

def load_image(image_path):
    """
    Load an image from file with error checking.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image or None if loading failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    
    return image

def calibrate_camera(objpoints, imgpoints, img_size):
    """
    Calibrate camera from object points and image points.
    
    Args:
        objpoints (list): List of arrays of 3D object points
        imgpoints (list): List of arrays of 2D image points
        img_size (tuple): Image size as (width, height)
        
    Returns:
        tuple: (camera_matrix, dist_coeffs, rvecs, tvecs) or (None, None, None, None) if calibration fails
    """
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    if not ret:
        print("Camera calibration failed.")
        return None, None, None, None
        
    return camera_matrix, dist_coeffs, rvecs, tvecs

def calculate_reprojection_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Calculate reprojection error from calibration parameters.
    
    Args:
        objpoints (list): List of arrays of 3D object points
        imgpoints (list): List of arrays of 2D image points
        camera_matrix (numpy.ndarray): Camera matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        rvecs (list): Rotation vectors for each image
        tvecs (list): Translation vectors for each image
        
    Returns:
        float: Mean reprojection error
    """
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        total_error += error
        
    return total_error / len(objpoints)

def prepare_object_points(pattern_size, square_size):
    """
    Prepare object points for camera calibration.
    
    Args:
        pattern_size (tuple): Pattern size as (columns, rows) of internal corners
        square_size (float): Size of chessboard squares in cm
        
    Returns:
        numpy.ndarray: Array of object points
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to real-world size
    return objp

def save_json(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        filename (str): Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def load_json(filename):
    """
    Load data from a JSON file.
    
    Args:
        filename (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data or None if loading failed
    """
    if not os.path.exists(filename):
        print(f"Error: JSON file {filename} not found.")
        return None
        
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def calculate_optimal_camera_matrix(camera_matrix, dist_coeffs, img_size):
    """
    Calculate optimal camera matrix for undistortion.
    
    Args:
        camera_matrix (numpy.ndarray): Camera matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        img_size (tuple): Image size as (width, height)
        
    Returns:
        tuple: (optimal_camera_matrix, roi)
    """
    return cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, img_size, 1, img_size)

def undistort_image_with_params(image, camera_matrix, dist_coeffs, optimal_camera_matrix=None):
    """
    Undistort an image using provided camera parameters.
    
    Args:
        image (numpy.ndarray): Image to undistort
        camera_matrix (numpy.ndarray): Camera matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        optimal_camera_matrix (numpy.ndarray): Optional optimal camera matrix
        
    Returns:
        numpy.ndarray: Undistorted image
    """
    if optimal_camera_matrix is None:
        h, w = image.shape[:2]
        optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
    
    # Undistort
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, optimal_camera_matrix)

def calculate_field_of_view(camera_matrix, img_size):
    """
    Calculate field of view from camera matrix.
    
    Args:
        camera_matrix (numpy.ndarray): Camera matrix
        img_size (tuple): Image size as (width, height)
        
    Returns:
        tuple: (fov_x, fov_y) in degrees
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    fov_x = 2 * np.arctan(img_size[0] / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(img_size[1] / (2 * fy)) * 180 / np.pi
    return fov_x, fov_y