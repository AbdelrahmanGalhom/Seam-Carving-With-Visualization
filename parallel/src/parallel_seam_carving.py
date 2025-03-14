import numpy as np
import cv2
from scipy.ndimage import convolve
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numba as nb
import matplotlib.pyplot as plt

class DynamicMapping:
    """
    Handles dynamic mapping of pixel positions in an image during seam carving.
    
    Attributes:
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        mapping (ndarray): 2D array where each position (r, c) in the reduced image
                           maps to a position in the original image.
    """
    
    def __init__(self, rows, cols):
        """
        Initialize the DynamicMapping object.
        
        Args:
            rows (int): Number of rows in the image.
            cols (int): Number of columns in the image.
        """
        self.rows = rows
        self.cols = cols
        
        # Initialize mapping: each position (r,c) in reduced image maps to (r,c) in original image
        self.mapping = np.arange(cols).reshape(1, cols).repeat(rows, axis=0)
    
    def apply_removals(self, removals):
        """
        Update the mapping after removing specified seams.
        
        Args:
            removals (list): List of (row, column) tuples indicating positions to remove.
        """
        for row, col in removals:
            if row < self.rows and col < self.cols:
                # Shift all mappings after the removed position
                for j in range(col, self.cols - 1):
                    self.mapping[row, j] = self.mapping[row, j + 1]
    
    def get_pos_in_original(self, row, col):
        """
        Get the position in the original image corresponding to the given position in the reduced image.
        
        Args:
            row (int): Row index in the reduced image.
            col (int): Column index in the reduced image.
            
        Returns:
            int: Column index in the original image.
        """
        if row < self.rows and col < self.mapping.shape[1]:
            return self.mapping[row, col]
        return col  # Fallback to using the same position if out of bounds


class ParallelSeamCarving:
    """
    Implements a parallelized version of the seam carving algorithm.
    """
    
    def __init__(self, image_path, target_height, target_width, n_workers=None):
        """
        Initialize the ParallelSeamCarving object.
        
        Args:
            image_path (str): Path to the input image.
            target_height (int): Desired height of the output image.
            target_width (int): Desired width of the output image.
            n_workers (int): Number of worker processes/threads. Defaults to CPU count.
        """
        self.image_path = image_path
        self.target_height = target_height
        self.target_width = target_width
        self.n_workers = n_workers or mp.cpu_count()
        
        # Load and initialize images
        self.input_image = cv2.imread(image_path).astype(np.float32)
        self.output_image = np.copy(self.input_image)
        self.visualization_image = np.copy(self.input_image)
        
        # Get image dimensions
        self.input_height, self.input_width = self.input_image.shape[:2]
        
        # Initialize pixel mapping using DynamicMapping class
        self.mapping = DynamicMapping(self.input_height, self.input_width)

    def process(self):
        """Execute the seam carving algorithm with parallelization."""
        # Calculate number of rows and columns to remove
        rows_to_remove = self.input_height - self.target_height
        cols_to_remove = self.input_width - self.target_width
        
        # Remove columns if needed (reduce width)
        if cols_to_remove > 0:
            self._remove_seams(cols_to_remove, is_vertical=True)
        
        # Remove rows if needed (reduce height)
        if rows_to_remove > 0:
            self.output_image = self._rotate_image(self.output_image, clockwise=False)
            self._remove_seams(rows_to_remove, is_vertical=False)
            self.output_image = self._rotate_image(self.output_image, clockwise=True)
    
    def _remove_seams(self, num_seams, is_vertical=True):
        """
        Remove the specified number of seams from the image.
        
        Args:
            num_seams (int): Number of seams to remove.
            is_vertical (bool): If True, remove vertical seams; otherwise, remove horizontal seams.
        """
        for _ in range(num_seams):
            # Calculate energy map using JIT compilation
            energy_map = self._calculate_energy_map_numba()
            
            # Find optimal seam using dynamic programming
            seam = self._find_optimal_seam_numba(energy_map)
            
            # Update visualization if removing vertical seams
            if is_vertical:
                self.visualization_image = self._visualize_seam(seam)
            
            # Get valid removals (within image bounds)
            valid_removals = []
            for row in range(len(seam)):
                if row < self.output_image.shape[0] and seam[row] < self.output_image.shape[1]:
                    valid_removals.append((row, seam[row]))
            
            # Update mapping
            self.mapping.apply_removals(valid_removals)
            
            # Remove seam
            self._delete_seam_numba(seam)
    
    def _calculate_energy_map(self):
        """
        Calculate the energy map using parallel processing.
        
        Returns:
            ndarray: Energy map of the image.
        """
        b, g, r = cv2.split(self.output_image)
        
        # Define kernels
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        # Parallelize energy calculation for each channel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            b_future = executor.submit(self._calc_channel_energy, b, kernel_x, kernel_y)
            g_future = executor.submit(self._calc_channel_energy, g, kernel_x, kernel_y)
            r_future = executor.submit(self._calc_channel_energy, r, kernel_x, kernel_y)
            
            b_energy = b_future.result()
            g_energy = g_future.result()
            r_energy = r_future.result()
        
        return b_energy + g_energy + r_energy
    
    def _calc_channel_energy(self, channel, kernel_x, kernel_y):
        """Calculate energy for a single channel."""
        grad_x = np.abs(convolve(channel, kernel_x, mode='reflect'))
        grad_y = np.abs(convolve(channel, kernel_y, mode='reflect'))
        return grad_x + grad_y
    
    @staticmethod
    @nb.njit(parallel=True)
    def _calculate_energy_map_static(image):
        """
        JIT-compiled function to calculate energy map.
        
        Args:
            image (ndarray): Input image.
            
        Returns:
            ndarray: Energy map.
        """
        height, width, _ = image.shape
        energy = np.zeros((height, width), dtype=np.float32)
        
        # Simple gradient calculation for efficiency
        for i in nb.prange(1, height-1):
            for j in nb.prange(1, width-1):
                for c in range(3):  # RGB channels
                    dx = float(image[i, j+1, c]) - float(image[i, j-1, c])
                    dy = float(image[i+1, j, c]) - float(image[i-1, j, c])
                    energy[i, j] += abs(dx) + abs(dy)
                    
        # Handle borders (simplified)
        energy[0, :] = energy[1, :]
        energy[-1, :] = energy[-2, :]
        energy[:, 0] = energy[:, 1]
        energy[:, -1] = energy[:, -2]
        
        return energy
    
    def _calculate_energy_map_numba(self):
        """Wrapper to call the static Numba function."""
        return ParallelSeamCarving._calculate_energy_map_static(self.output_image)
    
    @staticmethod
    @nb.njit
    def _find_optimal_seam_numba(energy_map):
        """
        JIT-compiled function to find the optimal seam.
        
        Args:
            energy_map (ndarray): Energy map.
            
        Returns:
            ndarray: Seam indices.
        """
        height, width = energy_map.shape
        cumulative_map = np.copy(energy_map)
        backtrack = np.zeros((height, width), dtype=np.int32)
        
        # Forward pass: build cumulative energy map
        for i in range(1, height):
            for j in range(width):
                # Check left neighbor
                if j > 0 and j < width-1:
                    # Handle all three cases: left-diagonal, up, right-diagonal
                    if cumulative_map[i-1, j-1] < cumulative_map[i-1, j] and cumulative_map[i-1, j-1] < cumulative_map[i-1, j+1]:
                        min_energy_col = j - 1
                    elif cumulative_map[i-1, j+1] < cumulative_map[i-1, j]:
                        min_energy_col = j + 1
                    else:
                        min_energy_col = j
                elif j == 0:
                    # Leftmost pixel: can only look up or right
                    if cumulative_map[i-1, j] < cumulative_map[i-1, j+1]:
                        min_energy_col = j
                    else:
                        min_energy_col = j + 1
                else:  # j == width-1
                    # Rightmost pixel: can only look up or left
                    if cumulative_map[i-1, j-1] < cumulative_map[i-1, j]:
                        min_energy_col = j - 1
                    else:
                        min_energy_col = j
                
                cumulative_map[i, j] += cumulative_map[i-1, min_energy_col]
                backtrack[i, j] = min_energy_col
        
        # Backward pass: find optimal path
        seam = np.zeros(height, dtype=np.int32)
        j = np.argmin(cumulative_map[-1])
        seam[-1] = j
        
        for i in range(height-2, -1, -1):
            j = backtrack[i+1, j]
            seam[i] = j
            
        return seam
    
    @staticmethod
    @nb.njit(parallel=True)
    def _delete_seam_static(image, seam):
        """
        JIT-compiled function to remove a seam from the image.
        
        Args:
            image (ndarray): Input image.
            seam (ndarray): Seam indices.
            
        Returns:
            ndarray: Image with seam removed.
        """
        height, width, channels = image.shape
        output = np.zeros((height, width-1, channels), dtype=image.dtype)
        
        for i in nb.prange(height):
            col = seam[i]
            for c in range(channels):
                # Copy pixels before the seam
                output[i, :col, c] = image[i, :col, c]
                # Copy pixels after the seam
                output[i, col:, c] = image[i, col+1:, c]
                
        return output
    
    def _delete_seam_numba(self, seam):
        """Wrapper to call the static Numba function and update the image."""
        self.output_image = ParallelSeamCarving._delete_seam_static(self.output_image, np.array(seam, dtype=np.int32))
    
    def _visualize_seam(self, seam):
        """
        Create a visualization of the seam on the input image.
        
        Args:
            seam (list): Indices of the seam.
            
        Returns:
            ndarray: Image with the seam highlighted.
        """
        height, width = self.visualization_image.shape[:2]
        output = np.copy(self.visualization_image)
        
        # Highlight seam pixels on the visualization image
        for row in range(min(height, len(seam))):
            if seam[row] < self.output_image.shape[1]:
                col = self.mapping.get_pos_in_original(row, seam[row])
                if 0 <= col < width:  # Ensure index is within bounds
                    output[row, col] = [0, 0, 255]  # Red color in BGR
        
        return output
    
    def _rotate_image(self, image, clockwise=True):
        """
        Rotate the image 90 degrees.
        
        Args:
            image (ndarray): Image to rotate.
            clockwise (bool): If True, rotate clockwise; otherwise, rotate counterclockwise.
            
        Returns:
            ndarray: Rotated image.
        """
        if clockwise:
            # Rotate 90 degrees clockwise
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Rotate 90 degrees counterclockwise
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def save_output_image(self):
        """Save the resized output image to disk."""
        img_name = self.image_path.split(".")[0]
        cv2.imwrite(f"{img_name}_output_parallel.jpg", self.output_image.astype(np.uint8))
    
    def save_visualization_image(self):
        """Save the visualization image to disk."""
        img_name = self.image_path.split(".")[0]
        cv2.imwrite(f"{img_name}_visualization_parallel.jpg", self.visualization_image.astype(np.uint8))

    def visualize_results(self):
        """Display input, output, and visualization images side by side."""
        # Convert images to uint8 for display
        input_img = self.input_image.astype(np.uint8)
        output_img = self.output_image.astype(np.uint8)
        vis_img = self.visualization_image.astype(np.uint8)
        
        # Create figure with three subplots
        plt.figure(figsize=(15, 5))
        
        # Input image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.axis('off')
        
        # Output image
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title("Output Image")
        plt.axis('off')

        # Visualization image
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("Visualization Image")
        plt.axis('off')

        plt.show()
        