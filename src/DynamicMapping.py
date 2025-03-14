import numpy as np

class DynamicMapping:
    """
    A class to handle dynamic mapping of pixel positions in an image during seam carving.
    Attributes:
    -----------
    rows : int
        Number of rows in the image.
    cols : int
        Number of columns in the image.
    mapping : numpy.ndarray
        A 2D array where each position (r, c) in the reduced image maps to position (r, c) in the original image.
    Methods:
    --------
    __init__(rows, cols):
        Initializes the DynamicMapping with the given number of rows and columns.
    apply_removals(removals):
        Updates the mapping after removing the specified seams.
    get_pos_in_original(r, c):
        Returns the position in the original image corresponding to the given position in the reduced image.
    """
    def __init__(self, rows, cols):
        """
        Initializes the DynamicMapping object with the given number of rows and columns.
        Args:
            rows (int): The number of rows in the image.
            cols (int): The number of columns in the image.
        Attributes:
            rows (int): The number of rows in the image.
            cols (int): The number of columns in the image.
            mapping (numpy.ndarray): A 2D array where each position (r, c) in the reduced image 
                                     maps to position (r, c) in the original image initially.
        """
        self.rows = rows
        self.cols = cols
        # Each position (r,c) in reduced image maps to position (r,c) in original image initially
        self.mapping = np.arange(cols).reshape(1, cols).repeat(rows, axis=0)
    
    def apply_removals(self, removals):
        """
        Apply the removals to the mapping by updating the mapping after removing the seam.

        Parameters:
        removals (list of tuples): A list of (row, column) tuples indicating the positions to be removed.

        Returns:
        None
        """
        # For each row, update the mapping after removing the seam
        for r, c in removals:
            if r < self.rows and c < self.cols:
                # Shift all mappings after the removed position
                for j in range(c, self.cols-1):
                    self.mapping[r, j] = self.mapping[r, j+1]
    
    def get_pos_in_original(self, r, c):
        """
        Get the position in the original image corresponding to the given row and column in the current image.

        Args:
            r (int): The row index in the current image.
            c (int): The column index in the current image.

        Returns:
            int: The column index in the original image if within bounds, otherwise the same column index.
        """
        if r < self.rows and c < self.mapping.shape[1]:
            return self.mapping[r, c]
        return c  # Fallback to using the same position if out of bounds
        
