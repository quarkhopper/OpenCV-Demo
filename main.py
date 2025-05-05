import argparse
from workflow import CalibrationWorkflow

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Chessboard corner detection and camera calibration')
    parser.add_argument('--square-size', type=float, default=0.8,
                      help='Size of each chessboard square in cm (default: 0.8 cm)')
    parser.add_argument('--pattern', type=str, default='7,7',
                      help='Pattern size as columns,rows (default: 7,7)')
    parser.add_argument('--ideal-dir', type=str, default='ideal_input_images',
                      help='Directory containing ideal reference images (default: ideal_input_images)')
    parser.add_argument('--experimental-dir', type=str, default='experimental_images',
                      help='Directory containing experimental images (default: experimental_images)')
    parser.add_argument('--generate-expected', '-g', action='store_true',
                      help='Generate expected.json with camera intrinsics and corner locations')
    parser.add_argument('--process-experimental', '-e', action='store_true',
                      help='Process experimental images and generate detection.json')
    parser.add_argument('--evaluate-quality', '-q', action='store_true',
                      help='Evaluate calibration quality and generate quality.txt')
    parser.add_argument('--generate-report', '-r', action='store_true',
                      help='Generate PDF report with image comparisons')
    parser.add_argument('--run-all', '-a', action='store_true',
                      help='Run the complete calibration workflow')
    parser.add_argument('--no-save-images', action='store_true',
                      help='Do not save intermediate calibration images')
    args = parser.parse_args()

    # Parse pattern size
    pattern_size = tuple(map(int, args.pattern.split(',')))
    
    # Create configuration for the workflow
    config = {
        "ideal_dir": args.ideal_dir,
        "experimental_dir": args.experimental_dir,
        "pattern_size": pattern_size,
        "square_size": args.square_size,
        "save_intermediate_images": not args.no_save_images
    }
    
    # Initialize the workflow
    workflow = CalibrationWorkflow(config)
    
    # Run the requested step
    if args.run_all:
        return 0 if workflow.run_full_workflow() else 1
    elif args.generate_expected:
        return 0 if workflow.generate_expected_json() else 1
    elif args.process_experimental:
        return 0 if workflow.process_experimental_images() else 1
    elif args.evaluate_quality:
        return 0 if workflow.evaluate_quality() else 1
    elif args.generate_report:
        return 0 if workflow.generate_report() else 1
    else:
        # Default behavior: print help
        parser.print_help()
        return 1

if __name__ == "__main__":
    exit(main())
