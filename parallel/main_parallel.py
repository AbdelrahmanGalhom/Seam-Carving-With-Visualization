# main_parallel.py
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from src.parallel_seam_carving import ParallelSeamCarving

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parallel Seam Carving for content-aware image resizing")
    parser.add_argument("--input", type=str, default="input.jpg", help="Input image filename")
    parser.add_argument("--height", type=int, required=True, help="Target output height")
    parser.add_argument("--width", type=int, required=True, help="Target output width")
    parser.add_argument("--workers", type=int, default=0, 
                        help="Number of worker processes/threads (0 = auto)")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare with non-parallelized version")
    parser.add_argument("--visualize", action="store_true", 
                        help="Display visualization after processing")
    
    return parser.parse_args()

def main():
    """Main function to execute parallel seam carving process."""
    args = parse_arguments()
    
    # Read the image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image '{args.input}'")
        return
    
    print(f"Input image shape: {image.shape}")
    
    # Execute parallel seam carving
    start_time = time.time()
    
    psc = ParallelSeamCarving(
        image_path=args.input,
        target_height=args.height,
        target_width=args.width,
        n_workers=args.workers if args.workers > 0 else None
    )
    
    psc.process()
    psc.save_output_image()
    psc.save_visualization_image()
    
    end_time = time.time()
    parallel_time = end_time - start_time
    print(f"Parallel execution time: {parallel_time:.2f} seconds")
    
    # Compare with regular version if requested
    if args.compare:
        from src.seam_carving import SeamCarving
        
        start_time = time.time()
        
        sc = SeamCarving(
            image_path=args.input,
            target_height=args.height,
            target_width=args.width
        )
        
        sc.process()
        
        end_time = time.time()
        regular_time = end_time - start_time
        print(f"Regular execution time: {regular_time:.2f} seconds")
        print(f"Speedup: {regular_time / parallel_time:.2f}x")
    
    print(f"Output image shape: {psc.output_image.shape}")
    
    # Display visualization if requested
    if args.visualize:
        psc.visualize_results()
    
    print("Done")

if __name__ == "__main__":
    main()