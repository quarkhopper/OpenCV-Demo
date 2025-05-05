import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import matplotlib.cm as cm
from quality import load_json_data

def draw_corners(image, corners, color, marker_type='cross', marker_size=10, thickness=3):
    """
    Draw corners on an image with specified color and size.
    
    Args:
        image (numpy.ndarray): Image on which to draw corners
        corners (list): List of [x,y] corner coordinates
        color (tuple): BGR color tuple for drawing
        marker_type (str): Type of marker ('cross' or 'box')
        marker_size (int): Size of marker in pixels
        thickness (int): Thickness of lines in pixels
        
    Returns:
        numpy.ndarray: Image with corners drawn
    """
    img_copy = image.copy()
    for corner in corners:
        x, y = int(corner[0]), int(corner[1])
        if marker_type == 'cross':
            cv2.drawMarker(
                img_copy, 
                (x, y), 
                color, 
                markerType=cv2.MARKER_CROSS, 
                markerSize=marker_size, 
                thickness=thickness
            )
        elif marker_type == 'box':
            half_size = marker_size // 2
            cv2.rectangle(
                img_copy, 
                (x - half_size, y - half_size),
                (x + half_size, y + half_size),
                color, 
                thickness
            )
    return img_copy

def create_undistortion_heatmap(image, camera_matrix, dist_coeffs):
    """
    Create a heatmap visualization of the undistortion applied to an image.
    
    Args:
        image (numpy.ndarray): Original image
        camera_matrix (numpy.ndarray): Camera matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        
    Returns:
        numpy.ndarray: Heatmap visualization of undistortion amount
    """
    h, w = image.shape[:2]
    
    # Create grid of points
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack((x.flatten(), y.flatten()), axis=1).astype(np.float32)
    
    # Undistort points
    undistorted_points = cv2.undistortPoints(
        points.reshape(-1, 1, 2), 
        camera_matrix, 
        dist_coeffs, 
        P=camera_matrix
    ).reshape(-1, 2)
    
    # Calculate displacement for each point (Euclidean distance)
    displacement = np.sqrt(np.sum((points - undistorted_points)**2, axis=1))
    
    # Reshape to image dimensions
    displacement_map = displacement.reshape(h, w)
    
    # Normalize for visualization
    max_displacement = np.max(displacement_map)
    if max_displacement > 0:
        norm_displacement = displacement_map / max_displacement
    else:
        norm_displacement = displacement_map
    
    # Create RGB heatmap
    heatmap = cm.jet(norm_displacement)[:,:,:3]  # Take only RGB channels
    
    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Create a blended version with original image
    alpha = 0.6  # Transparency of the heatmap
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(original_rgb, 1-alpha, heatmap_uint8, alpha, 0)
    
    return blended

def create_comparison_image(original_image, corners_ref, corners_exp, title=""):
    """
    Create a comparison image showing both reference and experimental corners.
    
    Args:
        original_image (numpy.ndarray): Original image
        corners_ref (list): List of [x,y] reference corner coordinates
        corners_exp (list): List of [x,y] experimental corner coordinates
        title (str): Title to draw on the image
        
    Returns:
        numpy.ndarray: Comparison image with both sets of corners
    """
    # Draw reference corners in green
    img_with_corners = draw_corners(original_image, corners_ref, (0, 255, 0), 12, 2)
    
    # Draw experimental corners in red
    img_with_corners = draw_corners(img_with_corners, corners_exp, (0, 0, 255), 8, 2)
    
    # Add title
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_with_corners, 
            title, 
            (10, 30), 
            font, 
            1, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
    
    return img_with_corners

def undistort_image_using_parameters(image, camera_matrix, dist_coeffs, optimal_camera_matrix=None):
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
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, optimal_camera_matrix)
    
    return undistorted

def recalculate_corners_after_undistortion(corners, camera_matrix, dist_coeffs, optimal_camera_matrix=None):
    """
    Recalculate corner positions after undistortion.
    
    Args:
        corners (list): List of [x,y] corner coordinates
        camera_matrix (numpy.ndarray): Camera matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        optimal_camera_matrix (numpy.ndarray): Optional optimal camera matrix
        
    Returns:
        list: Undistorted corner coordinates
    """
    corners_array = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    
    if optimal_camera_matrix is None:
        optimal_camera_matrix = camera_matrix
    
    # Undistort points
    undistorted_corners = cv2.undistortPoints(
        corners_array, 
        camera_matrix, 
        dist_coeffs,
        P=optimal_camera_matrix
    )
    
    return undistorted_corners.reshape(-1, 2).tolist()

def generate_visual_report(expected_json, detection_json, experimental_dir, output_pdf="calibration_report.pdf"):
    """
    Generate a comprehensive PDF report with visual comparisons.
    
    Args:
        expected_json (str): Path to the expected (reference) JSON file
        detection_json (str): Path to the detection (experimental) JSON file
        experimental_dir (str): Directory containing experimental images
        output_pdf (str): Path to save the PDF report
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load reference and experimental data
    reference_data = load_json_data(expected_json)
    detection_data = load_json_data(detection_json)
    
    if not reference_data or not detection_data:
        return False
        
    reference_corners = reference_data.get("corner_coordinates", {})
    detected_corners = detection_data.get("corner_coordinates", {})
    processing_status = detection_data.get("processing_status", {})
    all_images_processed = detection_data.get("all_images_processed", True)
    images_processed = detection_data.get("images_processed", 0)
    total_images = detection_data.get("total_images", 0)
    
    # Get camera parameters from detection data
    camera_matrix = np.array(detection_data["camera_matrix"]["raw"])
    dist_coeffs = np.array(detection_data["distortion_coefficients"])
    optimal_camera_matrix = np.array(detection_data["optimal_camera_matrix"])
    
    # Get all image names from reference data
    all_image_names = set(reference_corners.keys()).union(set(processing_status.keys()))
    
    # Create PDF
    with PdfPages(output_pdf) as pdf:
        # Title page with overall status
        plt.figure(figsize=(11, 8.5))
        plt.text(0.5, 0.85, 'Camera Calibration Report', 
                horizontalalignment='center', verticalalignment='center', fontsize=24)
        plt.text(0.5, 0.75, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        
        # Show overall status
        status_color = "green" if all_images_processed else "red"
        status_text = "PASS" if all_images_processed else "FAIL"
        plt.text(0.5, 0.65, f'Overall Status: {status_text}', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=18, color=status_color, weight='bold')
        
        # Show processing statistics
        plt.text(0.5, 0.55, f'Images Successfully Processed: {images_processed}/{total_images}', 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.text(0.5, 0.5, f'Reprojection Error: {detection_data["reprojection_error"]:.4f} pixels', 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Summary page with image processing status
        plt.figure(figsize=(11, 8.5))
        plt.text(0.5, 0.95, 'Image Processing Status', 
                horizontalalignment='center', verticalalignment='center', fontsize=18)
        
        # Create a table with image status
        status_data = [['Image', 'Status']]
        for img_name in sorted(all_image_names):
            status = processing_status.get(img_name, "Not processed")
            status_data.append([img_name, status])
        
        table = plt.table(cellText=status_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Color the status cells
        for i in range(1, len(status_data)):
            cell = table[(i, 1)]
            if "Success" in status_data[i][1]:
                cell.set_facecolor('#d8f3dc')  # Light green
            else:
                cell.set_facecolor('#ffccd5')  # Light red
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Process successful images for visualization
        for img_name in sorted(all_image_names):
            img_path = os.path.join(experimental_dir, img_name)
            
            # Handle failed images
            if img_name not in detected_corners:
                plt.figure(figsize=(11, 8.5))
                plt.suptitle(f"Image Processing Failed: {img_name}", fontsize=16, color='red')
                
                # Try to show the original image if available
                image = cv2.imread(img_path)
                if image is not None:
                    plt.subplot(1, 1, 1)
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.title(f"Original Image", fontsize=14)
                    plt.axis('off')
                    plt.figtext(0.5, 0.05, 
                                f"Status: {processing_status.get(img_name, 'Not processed')}", 
                                ha="center", fontsize=12, color="red", weight="bold")
                else:
                    plt.text(0.5, 0.5, f"Could not load image: {img_name}", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14)
                    plt.axis('off')
                
                pdf.savefig(bbox_inches='tight', pad_inches=0.5)
                plt.close()
                continue
                
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}, skipping visualization.")
                continue
                
            # Get corner data
            ref_corners = np.array(reference_corners[img_name])
            exp_corners = np.array(detected_corners[img_name])
            
            # Undistort the image
            undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, optimal_camera_matrix)
            
            # Create undistortion heatmap
            heatmap = create_undistortion_heatmap(image, camera_matrix, dist_coeffs)
            
            # Prepare images with corner overlays
            # Reference corners as crosses (blue)
            # Detected corners as boxes (green)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
            
            # Draw corners on original and undistorted images
            original_with_corners = draw_corners(image_rgb, ref_corners, (255, 0, 0), 'cross', 45, 3)  # Red crosses for ideal corners
            original_with_corners = draw_corners(original_with_corners, exp_corners, (0, 255, 0), 'box', 45, 3)  # Green boxes for detected corners
            
            # Calculate undistorted ref corners
            undistorted_ref_corners = np.array(ref_corners, dtype=np.float32).reshape(-1, 1, 2)
            undistorted_ref_corners = cv2.undistortPoints(
                undistorted_ref_corners, 
                camera_matrix, 
                dist_coeffs,
                P=optimal_camera_matrix
            ).reshape(-1, 2)
            
            # Calculate undistorted exp corners
            undistorted_exp_corners = np.array(exp_corners, dtype=np.float32).reshape(-1, 1, 2)
            undistorted_exp_corners = cv2.undistortPoints(
                undistorted_exp_corners, 
                camera_matrix, 
                dist_coeffs,
                P=optimal_camera_matrix
            ).reshape(-1, 2)
            
            # Draw corners on undistorted image
            undistorted_with_corners = draw_corners(undistorted_rgb, undistorted_ref_corners, (255, 0, 0), 'cross', 45, 3)  # Red crosses
            undistorted_with_corners = draw_corners(undistorted_with_corners, undistorted_exp_corners, (0, 255, 0), 'box', 45, 3)  # Green boxes
            
            # Create figure with image visualizations
            plt.figure(figsize=(11, 8.5))
            plt.suptitle(f"Image Analysis: {img_name}", fontsize=16)
            
            # Show original image with corners
            plt.subplot(1, 3, 1)
            plt.imshow(original_with_corners)
            plt.title("Original Image", fontsize=12)
            plt.axis('off')
            
            # Show undistorted image with corners
            plt.subplot(1, 3, 2)
            plt.imshow(undistorted_with_corners)
            plt.title("Undistorted Image", fontsize=12)
            plt.axis('off')
            
            # Show undistortion heatmap
            plt.subplot(1, 3, 3)
            plt.imshow(heatmap)
            plt.title("Undistortion Heatmap", fontsize=12)
            plt.axis('off')
            
            # Add corner color legend
            plt.figtext(0.5, 0.04, 
                      "Red Crosses: Ideal Corner Locations    Green Boxes: Detected Corner Locations", 
                      ha="center", fontsize=10)
            
            # Add quality stats for this image
            if img_name in detection_data.get("quality_stats", {}):
                stats = detection_data["quality_stats"][img_name]
                stats_text = f"Max Deviation: {stats['max_deviation']:.2f} px   " \
                           f"Mean Deviation: {stats['mean_deviation']:.2f} px   " \
                           f"Std Deviation: {stats['std_deviation']:.2f} px"
                plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            pdf.savefig()
            plt.close()
        
        # Final page with camera parameters
        plt.figure(figsize=(11, 8.5))
        plt.suptitle('Camera Calibration Parameters', fontsize=20)
        
        # Create camera matrix table
        plt.subplot(2, 2, 1)
        plt.title("Camera Matrix", fontsize=14)
        camera_matrix_rounded = np.round(camera_matrix, 2)
        plt.table(
            cellText=camera_matrix_rounded,
            loc='center',
            cellLoc='center'
        )
        plt.axis('off')
        
        # Create distortion coefficients table
        plt.subplot(2, 2, 2)
        plt.title("Distortion Coefficients", fontsize=14)
        dist_names = ["k1", "k2", "p1", "p2", "k3"]
        dist_values = [[f"{val:.6f}"] for val in dist_coeffs.flatten()]
        dist_table = plt.table(
            cellText=dist_values,
            rowLabels=dist_names,
            loc='center',
            cellLoc='center'
        )
        dist_table.auto_set_font_size(False)
        dist_table.set_fontsize(10)
        dist_table.scale(1, 1.5)
        plt.axis('off')
        
        # Add field of view information
        plt.subplot(2, 2, 3)
        plt.title("Field of View", fontsize=14)
        fov_horizontal = detection_data["field_of_view"]["horizontal"]
        fov_vertical = detection_data["field_of_view"]["vertical"]
        fov_data = [
            [f"{fov_horizontal:.2f}°"],
            [f"{fov_vertical:.2f}°"]
        ]
        fov_table = plt.table(
            cellText=fov_data,
            rowLabels=["Horizontal", "Vertical"],
            loc='center',
            cellLoc='center'
        )
        fov_table.auto_set_font_size(False)
        fov_table.set_fontsize(10)
        fov_table.scale(1, 1.5)
        plt.axis('off')
        
        # Add reprojection error
        plt.subplot(2, 2, 4)
        plt.title("Calibration Quality", fontsize=14)
        error = detection_data["reprojection_error"]
        error_table = plt.table(
            cellText=[[f"{error:.6f} pixels"]],
            rowLabels=["Reprojection Error"],
            loc='center',
            cellLoc='center'
        )
        error_table.auto_set_font_size(False)
        error_table.set_fontsize(10)
        error_table.scale(1, 1.5)
        plt.axis('off')
        
        # Add additional info
        plt.figtext(0.5, 0.05, 
                  f"Image Resolution: {detection_data['image_size'][0]}x{detection_data['image_size'][1]} pixels", 
                  ha="center", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])
        pdf.savefig()
        plt.close()
    
    print(f"Visual report saved to {output_pdf}")
    return True

def generate_detection_json(calibration_data, output_json="detection.json"):
    """
    Save calibration data to a JSON file.
    
    Args:
        calibration_data (dict): Calibration data from process_experimental_images
        output_json (str): Path to save the JSON output
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not calibration_data:
        print("Error: No calibration data to save")
        return False
        
    try:
        with open(output_json, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Detection data saved to {output_json}")
        return True
    except Exception as e:
        print(f"Error saving detection JSON: {e}")
        return False