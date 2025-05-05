import os
import numpy as np
import utils
from quality import process_experimental_images, evaluate_calibration_quality, generate_quality_report
from reports import generate_detection_json, generate_visual_report

class CalibrationWorkflow:
    """
    Manages the camera calibration workflow process.
    """
    
    def __init__(self, config=None):
        """
        Initialize the workflow with configuration.
        
        Args:
            config (dict): Configuration dictionary with settings
        """
        # Default configuration
        self.config = {
            "ideal_dir": "ideal_input_images",
            "experimental_dir": "experimental_images",
            "expected_json": "expected.json",
            "detection_json": "detection.json",
            "quality_txt": "quality.txt",
            "report_pdf": "calibration_report.pdf",
            "image_names": ["A.png", "B.png", "C.png", "D.png"],
            "pattern_size": (7, 7),
            "square_size": 0.8,
            "save_intermediate_images": True
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
    
    def validate_directory(self, directory, image_names=None):
        """
        Validate that a directory exists and contains required images.
        
        Args:
            directory (str): Directory path to validate
            image_names (list): Optional list of image filenames to check for
            
        Returns:
            tuple: (success, missing_images)
        """
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found.")
            return False, []
            
        if image_names:
            image_paths = [os.path.join(directory, name) for name in image_names]
            missing = [img for img in image_paths if not os.path.exists(img)]
            if missing:
                print(f"Error: Missing images: {[os.path.basename(img) for img in missing]}")
                return False, missing
        
        return True, []
    
    def generate_expected_json(self):
        """
        Generate expected.json from ideal input images.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate directory and images
        ideal_dir = self.config["ideal_dir"]
        image_names = self.config["image_names"]
        success, missing = self.validate_directory(ideal_dir, image_names)
        if not success:
            return False
            
        # Get image paths
        ideal_images = [os.path.join(ideal_dir, name) for name in image_names]
        print(f"Processing {len(ideal_images)} ideal images...")
        
        # Process images and collect corner data
        objpoints = []
        imgpoints = []
        corner_coords = {}
        pattern_size = self.config["pattern_size"]
        square_size = self.config["square_size"]
        objp = utils.prepare_object_points(pattern_size, square_size)
        
        # Process each image
        for img_path in ideal_images:
            img_name = os.path.basename(img_path)
            print(f"Processing {img_name}...")
            
            # Load image
            img = utils.load_image(img_path)
            if img is None:
                return False
                
            # Find corners
            ret, corners = utils.find_chessboard_corners(img, pattern_size)
            if not ret:
                print(f"Error: Could not find corners in {img_path}")
                return False
                
            # Store points
            objpoints.append(objp)
            imgpoints.append(corners)
            corner_coords[img_name] = corners.reshape(-1, 2).tolist()
        
        # Calibrate camera
        img_size = img.shape[1::-1]  # (width, height)
        camera_matrix, dist_coeffs, rvecs, tvecs = utils.calibrate_camera(
            objpoints, imgpoints, img_size
        )
        
        if camera_matrix is None:
            print("Calibration failed.")
            return False
            
        # Calculate reprojection error
        mean_error = utils.calculate_reprojection_error(
            objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs
        )
        
        # Calculate field of view
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        fov_x, fov_y = utils.calculate_field_of_view(camera_matrix, img_size)
        
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
            "image_size": img_size,
            "reprojection_error": float(mean_error),
            "field_of_view": {
                "horizontal": float(fov_x),
                "vertical": float(fov_y)
            },
            "corner_coordinates": corner_coords
        }
        
        # Save to expected.json
        output_file = self.config["expected_json"]
        if utils.save_json(calibration_data, output_file):
            print(f"Generated {output_file} with reference calibration data.")
            return True
        else:
            return False
    
    def process_experimental_images(self):
        """
        Process experimental images and generate detection.json.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate directory and images
        exp_dir = self.config["experimental_dir"]
        image_names = self.config["image_names"]
        success, missing = self.validate_directory(exp_dir, image_names)
        if not success:
            return False
            
        # Get image paths
        experimental_images = [os.path.join(exp_dir, name) for name in image_names]
        
        # Process images and generate detection.json
        pattern_size = self.config["pattern_size"]
        square_size = self.config["square_size"]
        calibration_data = process_experimental_images(
            experimental_images, 
            pattern_size, 
            square_size
        )
        
        if calibration_data is None:
            print("Critical failure in processing experimental images.")
            return False
            
        # Save to detection.json
        output_file = self.config["detection_json"]
        if generate_detection_json(calibration_data, output_file):
            if calibration_data.get("all_images_processed", True):
                print(f"Generated {output_file} with experimental calibration data.")
            else:
                images_processed = calibration_data.get("images_processed", 0)
                total_images = calibration_data.get("total_images", 0)
                print(f"Warning: Only {images_processed}/{total_images} images were successfully processed.")
                print(f"Partial results saved to {output_file}.")
                
                # Print processing status for each image
                for img_name, status in calibration_data.get("processing_status", {}).items():
                    print(f"  {img_name}: {status}")
                
            return True
        else:
            return False
    
    def evaluate_quality(self):
        """
        Evaluate calibration quality and generate quality.txt.
        
        Returns:
            bool: True if successful, False otherwise
        """
        expected_json = self.config["expected_json"]
        detection_json = self.config["detection_json"]
        
        # Check that required files exist
        if not os.path.exists(expected_json):
            print(f"Error: {expected_json} not found. Run generate_expected_json first.")
            return False
            
        if not os.path.exists(detection_json):
            print(f"Error: {detection_json} not found. Run process_experimental_images first.")
            return False
            
        # Evaluate quality by comparing corner positions
        quality_data = evaluate_calibration_quality(expected_json, detection_json)
        if quality_data is None:
            print("Failed to evaluate calibration quality.")
            return False
            
        # Generate quality.txt report
        quality_txt = self.config["quality_txt"]
        if generate_quality_report(quality_data, quality_txt):
            print(f"Generated {quality_txt} with calibration quality assessment.")
            return True
        else:
            return False
    
    def generate_report(self):
        """
        Generate PDF report with image comparisons.
        
        Returns:
            bool: True if successful, False otherwise
        """
        expected_json = self.config["expected_json"]
        detection_json = self.config["detection_json"]
        exp_dir = self.config["experimental_dir"]
        report_pdf = self.config["report_pdf"]
        
        # Check that required files exist
        if not os.path.exists(expected_json):
            print(f"Error: {expected_json} not found. Run generate_expected_json first.")
            return False
            
        if not os.path.exists(detection_json):
            print(f"Error: {detection_json} not found. Run process_experimental_images first.")
            return False
            
        # Generate PDF report
        if generate_visual_report(expected_json, detection_json, exp_dir, report_pdf):
            print(f"Generated {report_pdf} with visual comparisons.")
            return True
        else:
            return False
    
    def run_full_workflow(self):
        """
        Run the complete calibration workflow sequence.
        
        Returns:
            bool: True if successful, False otherwise
        """
        steps = [
            ("Generating expected.json", self.generate_expected_json),
            ("Processing experimental images", self.process_experimental_images),
            ("Evaluating calibration quality", self.evaluate_quality),
            ("Generating visual report", self.generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n=== {step_name} ===")
            success = step_func()
            if not success:
                print(f"Workflow failed at step: {step_name}")
                return False
        
        print("\nWorkflow completed successfully!")
        return True