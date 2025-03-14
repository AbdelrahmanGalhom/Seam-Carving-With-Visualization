import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve
from .DynamicMapping import DynamicMapping

class SeamCarving:
    """
    Implements the seam carving algorithm for content-aware image resizing.
    
    Attributes:
        image_path (str): Path to the input image.
        target_height (int): Desired height of the output image.
        target_width (int): Desired width of the output image.
        input_image (ndarray): Original input image.
        output_image (ndarray): Resized output image.
        visualization_image (ndarray): Image with seam visualization.
        mapping (DynamicMapping): Tracks pixel positions between original and resized images.
    """
    
    def __init__(self, image_path, target_height, target_width):
        """
        Initialize the SeamCarving object.
        
        Args:
            image_path (str): Path to the input image.
            target_height (int): Desired height of the output image.
            target_width (int): Desired width of the output image.
        """
        self.image_path = image_path
        self.target_height = target_height
        self.target_width = target_width
        
        # Load and initialize images
        self.input_image = cv2.imread(image_path).astype(np.float32)
        self.output_image = np.copy(self.input_image)
        self.visualization_image = np.copy(self.input_image)
        
        # Get image dimensions
        self.input_height, self.input_width = self.input_image.shape[:2]
        
        # Initialize pixel mapping
        self.mapping = DynamicMapping(self.input_height, self.input_width)

    def process(self):
        """Execute the seam carving algorithm to resize the image."""
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
            # Calculate energy map
            energy_map = self._calculate_energy_map()
            
            # Find optimal seam using dynamic programming
            seam = self._find_optimal_seam(energy_map)
            
            # Update visualization if removing vertical seams
            if is_vertical:
                self.visualization_image = self._visualize_seam(seam)
            
            # Get valid removals (within image bounds)
            valid_removals = []
            for row in range(len(seam)):
                if row < self.output_image.shape[0] and seam[row] < self.output_image.shape[1]:
                    valid_removals.append((row, seam[row]))
            
            # Update mapping and remove seam
            self.mapping.apply_removals(valid_removals)
            self._delete_seam(seam)
    
    def _calculate_energy_map(self):
        """
        Calculate the energy map of the current image.
        
        Returns:
            ndarray: Energy map of the image.
        """
        b, g, r = cv2.split(self.output_image)
        
        # Using Sobel-like kernels for energy calculation
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        # Calculate energy for each channel
        b_energy = np.abs(convolve(b, kernel_x, mode='reflect')) + np.abs(convolve(b, kernel_y, mode='reflect'))
        g_energy = np.abs(convolve(g, kernel_x, mode='reflect')) + np.abs(convolve(g, kernel_y, mode='reflect'))
        r_energy = np.abs(convolve(r, kernel_x, mode='reflect')) + np.abs(convolve(r, kernel_y, mode='reflect'))
        
        # Combine channel energies
        return b_energy + g_energy + r_energy
    
    def _find_optimal_seam(self, energy_map):
        """
        Find the optimal seam using dynamic programming.
        
        Args:
            energy_map (ndarray): Energy map of the image.
            
        Returns:
            list: Indices of the optimal seam.
        """
        height, width = energy_map.shape
        
        # Initialize cumulative energy map and backtracking array
        cumulative_map = energy_map.copy()
        backtrack = np.zeros_like(cumulative_map, dtype=np.int32)
        
        # Dynamic programming to find minimum energy path
        for i in range(1, height):
            for j in range(width):
                # Check left neighbor
                if j > 0 and cumulative_map[i-1, j-1] < cumulative_map[i-1, j]:
                    min_energy_col = j - 1
                else:
                    min_energy_col = j
                
                # Check right neighbor
                if j < width-1 and cumulative_map[i-1, j+1] < cumulative_map[i-1, min_energy_col]:
                    min_energy_col = j + 1
                
                # Update cumulative energy and backtrack arrays
                cumulative_map[i, j] += cumulative_map[i-1, min_energy_col]
                backtrack[i, j] = min_energy_col
        
        # Backtrack to find the seam
        seam = []
        min_energy_col = np.argmin(cumulative_map[-1])
        
        for i in range(height-1, -1, -1):
            seam.append(min_energy_col)
            min_energy_col = backtrack[i, min_energy_col]
            
        return seam[::-1]  # Reverse to get top-to-bottom order
    
    def _delete_seam(self, seam):
        """
        Remove the specified seam from the output image.
        
        Args:
            seam (list): Indices of the seam to remove.
        """
        height, width = self.output_image.shape[:2]
        new_image = np.zeros((height, width - 1, 3))
        
        for row in range(height):
            col = seam[row]
            new_image[row, :, 0] = np.delete(self.output_image[row, :, 0], col)
            new_image[row, :, 1] = np.delete(self.output_image[row, :, 1], col)
            new_image[row, :, 2] = np.delete(self.output_image[row, :, 2], col)
            
        self.output_image = new_image
    
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
        height, width, channels = image.shape
        rotated = np.zeros((width, height, channels))
        
        if not clockwise:  # Counterclockwise rotation
            flipped = np.fliplr(image)
            for c in range(channels):
                for row in range(height):
                    rotated[:, row, c] = flipped[row, :, c]
        else:  # Clockwise rotation
            for c in range(channels):
                for row in range(height):
                    rotated[:, height - 1 - row, c] = image[row, :, c]
                    
        return rotated
    
    def save_output_image(self):
        """Save the resized output image to disk."""
        img_name = self.image_path.split(".")[0]
        cv2.imwrite(f"{img_name}_output.jpg", self.output_image)
    
    def save_visualization_image(self):
        """Save the visualization image to disk."""
        img_name = self.image_path.split(".")[0]
        cv2.imwrite(f"{img_name}_visualization.jpg", self.visualization_image)
    
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
        plt.title("Seams Visualization")
        plt.axis('off')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
