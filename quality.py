import numpy as np
import cv2
import json
import os
from datetime import datetime
import statistics

def load_json_data(json_file):
    """
    Load calibration data from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data or None if file doesn't exist
    """
    if not os.path.exists(json_file):
        print(f"Error: JSON file {json_file} not found.")
        return None
        
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def calculate_deviations(reference_corners, detected_corners):
    """
    Calculate pixel deviations between reference and detected corners
    by finding the closest reference corner for each detected corner.
    
    Args:
        reference_corners (list): List of [x,y] corner coordinates from reference
        detected_corners (list): List of [x,y] corner coordinates from detection
        
    Returns:
        dict: Dictionary with deviation statistics
    """
    if not reference_corners or not detected_corners:
        print("Error: Empty corner set provided")
        return None
    
    # For each detected corner, find the closest reference corner
    distances = []
    closest_ref_indices = []  # Keep track of which reference corner is closest to each detected corner
    
    for det_corner in detected_corners:
        # Calculate distance to each reference corner
        corner_distances = []
        for ref_corner in reference_corners:
            dist = np.sqrt((ref_corner[0] - det_corner[0])**2 + (ref_corner[1] - det_corner[1])**2)
            corner_distances.append(dist)
        
        # Find the closest reference corner
        min_dist_idx = np.argmin(corner_distances)
        min_dist = corner_distances[min_dist_idx]
        
        distances.append(min_dist)
        closest_ref_indices.append(min_dist_idx)
    
    # Calculate statistics
    max_deviation = max(distances)
    mean_deviation = sum(distances) / len(distances)
    std_deviation = statistics.stdev(distances) if len(distances) > 1 else 0
    
    return {
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation, 
        "std_deviation": std_deviation,
        "distances": distances,
        "closest_ref_indices": closest_ref_indices
    }

def evaluate_calibration_quality(expected_json, detection_json):
    """
    Evaluate calibration quality by comparing corner positions.
    
    Args:
        expected_json (str): Path to the expected (reference) JSON file
        detection_json (str): Path to the detection (experimental) JSON file
        
    Returns:
        dict: Dictionary with quality metrics for each image and overall
    """
    # Load reference and experimental data
    reference_data = load_json_data(expected_json)
    detection_data = load_json_data(detection_json)
    
    if not reference_data or not detection_data:
        return None
        
    reference_corners = reference_data.get("corner_coordinates", {})
    detected_corners = detection_data.get("corner_coordinates", {})
    
    # Validate that we have matching images
    common_images = set(reference_corners.keys()).intersection(set(detected_corners.keys()))
    if not common_images:
        print("Error: No matching images found between reference and detection data")
        return None
    
    # Calculate deviations for each image
    quality_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_image_stats": {},
        "overall_stats": {
            "max_deviation": 0,
            "mean_deviation": 0,
            "std_deviation": 0
        }
    }
    
    all_distances = []
    
    for img_name in common_images:
        ref_corners = reference_corners[img_name]
        det_corners = detected_corners[img_name]
        
        deviations = calculate_deviations(ref_corners, det_corners)
        if deviations:
            quality_data["per_image_stats"][img_name] = deviations
            all_distances.extend(deviations["distances"])
    
    # Calculate overall statistics
    if all_distances:
        quality_data["overall_stats"]["max_deviation"] = max(all_distances)
        quality_data["overall_stats"]["mean_deviation"] = sum(all_distances) / len(all_distances)
        quality_data["overall_stats"]["std_deviation"] = statistics.stdev(all_distances) if len(all_distances) > 1 else 0
    
    # Also add quality stats to the detection data for use in the PDF report
    detection_data["quality_stats"] = {
        img: {
            "max_deviation": stats["max_deviation"],
            "mean_deviation": stats["mean_deviation"],
            "std_deviation": stats["std_deviation"]
        }
        for img, stats in quality_data["per_image_stats"].items()
    }
    
    # Save the updated detection data
    try:
        with open(detection_json, 'w') as f:
            json.dump(detection_data, f, indent=2)
        print(f"Updated {detection_json} with quality statistics")
    except Exception as e:
        print(f"Warning: Could not update detection JSON with quality stats: {e}")
    
    return quality_data

def generate_quality_report(quality_data, output_file="quality.txt"):
    """
    Generate a text quality report from quality metrics.
    
    Args:
        quality_data (dict): Quality metrics from evaluate_calibration_quality
        output_file (str): Path to save the report
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not quality_data:
        print("Error: No quality data to generate report")
        return False
        
    try:
        with open(output_file, 'w') as f:
            f.write("Camera Calibration Quality Assessment Report\n")
            f.write("==========================================\n\n")
            f.write(f"Date: {quality_data['date']}\n\n")
            
            f.write("Per-Image Statistics:\n")
            f.write("--------------------\n\n")
            
            for img_name, stats in quality_data["per_image_stats"].items():
                f.write(f"Image: {img_name}\n")
                f.write(f"  Max deviation: {stats['max_deviation']:.4f} pixels\n")
                f.write(f"  Mean deviation: {stats['mean_deviation']:.4f} pixels\n")
                f.write(f"  Standard deviation: {stats['std_deviation']:.4f} pixels\n\n")
            
            f.write("Overall Statistics:\n")
            f.write("-----------------\n\n")
            f.write(f"  Max deviation across all images: {quality_data['overall_stats']['max_deviation']:.4f} pixels\n")
            f.write(f"  Mean deviation across all images: {quality_data['overall_stats']['mean_deviation']:.4f} pixels\n")
            f.write(f"  Standard deviation across all images: {quality_data['overall_stats']['std_deviation']:.4f} pixels\n")
            
        print(f"Quality report saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error generating quality report: {e}")
        return False

def process_experimental_images(image_paths, pattern_size=(7, 7), square_size=0.8):
    """
    Process experimental images to detect corners and calculate camera parameters.
    
    Args:
        image_paths (list): List of paths to experimental images
        pattern_size (tuple): Chessboard pattern size (columns, rows)
        square_size (float): Size of chessboard square in cm
        
    Returns:
        dict: Calibration data including corner coordinates, or None if critical failure
    """
    # Initialize arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    # Dictionary to store corner coordinates for each image
    corner_coords = {}
    
    # Dictionary to store processing status for each image
    processing_status = {}
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (cols-1,rows-1,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Scale object points to real-world dimensions (in cm)
    objp *= square_size
    
    # Get image size from first available image
    img_size = None
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img_size = (img.shape[1], img.shape[0])
            break
    
    if img_size is None:
        print("Error: Could not load any images")
        return None
    
    # Process each image
    print(f"Processing {len(image_paths)} experimental images...")
    all_success = True
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"Processing image {idx+1}/{len(image_paths)}: {img_name}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Error: Could not load image {img_path}, skipping.")
            processing_status[img_name] = "Failed: Could not load image"
            all_success = False
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add object and image points
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Store corner coordinates for this image
            corner_coords[img_name] = corners.reshape(-1, 2).tolist()
            processing_status[img_name] = "Success"
        else:
            print(f"  Warning: No corners found in {img_path}")
            processing_status[img_name] = "Failed: No corners detected"
            all_success = False
    
    # Check if we have at least one successful image
    if not objpoints:
        print("Error: No usable images found. Cannot calibrate.")
        return None
    
    # Calibrate camera with available images
    print(f"Calibrating camera from {len(objpoints)} images...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    if not ret:
        print("Calibration failed.")
        return None
        
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        total_error += error
        
    mean_error = total_error / len(objpoints)
    
    # Print calibration results
    print("\nExperimental Camera Calibration Results:")
    print(f"Calibrated with {len(objpoints)}/{len(image_paths)} images")
    print(f"Reprojection error: {mean_error}")
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist_coeffs)
    
    # Calculate field of view
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    fov_x = 2 * np.arctan(img_size[0] / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(img_size[1] / (2 * fy)) * 180 / np.pi
    
    # Calculate optimal camera matrix for undistortion
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, 1, img_size
    )
    
    # Prepare calibration data for JSON
    calibration_data = {
        "camera_matrix": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "raw": camera_matrix.tolist()
        },
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "optimal_camera_matrix": optimal_camera_matrix.tolist(),
        "roi": [int(x) for x in roi],
        "image_size": img_size,
        "reprojection_error": float(mean_error),
        "field_of_view": {
            "horizontal": float(fov_x),
            "vertical": float(fov_y)
        },
        "corner_coordinates": corner_coords,
        "processing_status": processing_status,
        "all_images_processed": all_success,
        "images_processed": len(objpoints),
        "total_images": len(image_paths)
    }
    
    return calibration_data