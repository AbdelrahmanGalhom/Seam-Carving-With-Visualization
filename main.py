import cv2
import time
import argparse
from src.SeamCarving import SeamCarving 

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Seam Carving for content-aware image resizing")
    parser.add_argument("--input", type=str, default="input.jpg", help="Input image filename")
    parser.add_argument("--height", type=int, required=True, help="Target output height")
    parser.add_argument("--width", type=int, required=True, help="Target output width")
    parser.add_argument("--visualize", action="store_true", help="Display visualization after processing")
    return parser.parse_args()

def main():
    """Main function to execute seam carving process."""
    args = parse_arguments()
    
    # Read the image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image '{args.input}'")
        return
    
    print(f"Input image shape: {image.shape}")
    
    # Execute seam carving
    start_time = time.time()
    
    sc = SeamCarving(
        image_path=args.input,
        target_height=args.height,
        target_width=args.width
    )
    
    sc.process()
    sc.save_output_image()
    sc.save_visualization_image()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    print(f"Output image shape: {sc.output_image.shape}")
    
    # Display visualization if requested
    if args.visualize:
        sc.visualize_results()
    
    print("Done")

if __name__ == "__main__":
    main()
