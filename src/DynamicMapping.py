import numpy as np

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
