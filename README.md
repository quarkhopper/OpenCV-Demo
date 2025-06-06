# OpenCV Camera Calibration Tool

A robust Python tool for camera calibration using chessboard patterns, built with OpenCV. This tool processes images of chessboard patterns to calculate camera intrinsic parameters, evaluate calibration quality, and generate comprehensive reports.

## Features

- Chessboard corner detection in images
- Camera calibration with intrinsic parameter calculation
- Quality assessment of calibration results
- PDF report generation with visual comparisons
- Resilient processing that continues even when some images fail
- Undistortion visualization with heatmaps

## Requirements

- Python 3.6 or higher
- OpenCV 4.x
- NumPy
- Matplotlib
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/quarkhopper/OpenCV-Demo.git
cd OpenCV-Demo
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

To run the complete calibration workflow with default settings:

```bash
python main.py --run-all --experimental-dir your_images_folder
```

### Input Images

The tool requires two sets of images:

1. **Reference (ideal) images**: Chessboard patterns from a known, calibrated setup
   - Default location: `ideal_input_images/`

2. **Experimental images**: Chessboard patterns captured with the camera you want to calibrate
   - Default location: `experimental_images/` (specify with `--experimental-dir`)

### Command Line Options

```
usage: main.py [-h] [--square-size SQUARE_SIZE] [--pattern PATTERN]
               [--ideal-dir IDEAL_DIR] [--experimental-dir EXPERIMENTAL_DIR]
               [--generate-expected] [--process-experimental] [--evaluate-quality]
               [--generate-report] [--run-all] [--no-save-images]

Chessboard corner detection and camera calibration

optional arguments:
  -h, --help            Show this help message and exit
  --square-size SQUARE_SIZE
                        Size of each chessboard square in cm (default: 0.8 cm)
  --pattern PATTERN     Pattern size as columns,rows (default: 7,7)
  --ideal-dir IDEAL_DIR
                        Directory containing ideal reference images (default: ideal_input_images)
  --experimental-dir EXPERIMENTAL_DIR
                        Directory containing experimental images (default: experimental_images)
  --generate-expected, -g
                        Generate expected.json with camera intrinsics and corner locations
  --process-experimental, -e
                        Process experimental images and generate detection.json
  --evaluate-quality, -q
                        Evaluate calibration quality and generate quality.txt
  --generate-report, -r
                        Generate PDF report with image comparisons
  --run-all, -a         Run the complete calibration workflow
  --no-save-images      Do not save intermediate calibration images
```

### Step-by-Step Workflow

You can run the complete workflow or individual steps:

#### 1. Generate reference data

```bash
python main.py --generate-expected
```
This processes reference images and generates `expected.json` with corner coordinates.

#### 2. Process experimental images

```bash
python main.py --process-experimental --experimental-dir your_images_folder
```
This detects corners in experimental images, calibrates the camera, and saves results to `detection.json`.

#### 3. Evaluate calibration quality

```bash
python main.py --evaluate-quality
```
This compares reference and detected corners to evaluate calibration quality, saving results to `quality.txt`.

#### 4. Generate visual report

```bash
python main.py --generate-report
```
This creates `calibration_report.pdf` with visual comparisons and statistics.

### Output Files

- `expected.json`: Camera parameters and corner locations from reference images
- `detection.json`: Camera parameters and corner locations from experimental images
- `quality.txt`: Calibration quality statistics
- `calibration_report.pdf`: Visual report with corner detection, undistortion, and quality metrics
- `camera_calibration.npz`: Camera parameters in NumPy format (optional)

## Report Visualization

The PDF report includes:
- Summary of processed images and overall calibration status
- For each successful image:
  - Original image with corners marked (red crosses for ideal corners, green boxes for detected corners)
  - Undistorted image with corrected corners
  - Undistortion heatmap showing distortion correction intensity
  - Quality statistics for each image
- Camera parameters and calibration metrics on the final page

## Error Handling

The tool is designed to be robust and will:
- Continue processing when some images fail corner detection
- Provide partial results with clear indications of which images failed
- Include warning messages in the quality report when not all images were processed

## Example

To calibrate your camera using chessboard images:

1. Print a 7x7 chessboard pattern
2. Take several photos of the chessboard from different angles
3. Place photos in a directory (e.g., `my_camera_images/`)
4. Run:
   ```bash
   python main.py --run-all --experimental-dir my_camera_images
   ```
5. Review the generated `calibration_report.pdf` for results

## Advanced Customization

For different chessboard patterns:

```bash
python main.py --run-all --pattern 9,6 --square-size 2.5 --experimental-dir my_images
```
This uses a 9x6 chessboard with 2.5cm squares.

## Troubleshooting

- **No corners detected**: Make sure the entire chessboard is visible with good lighting and contrast
- **High reprojection error**: Try with more images or improve image quality
- **Inconsistent undistortion**: Check if your lens requires a more complex distortion model

## License

This project is licensed under the MIT License - see the LICENSE file for details.
