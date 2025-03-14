import numpy as np

class DynamicMapping:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # Each position (r,c) in reduced image maps to position (r,c) in original image initially
        self.mapping = np.arange(cols).reshape(1, cols).repeat(rows, axis=0)
    
    def apply_removals(self, removals):
        # For each row, update the mapping after removing the seam
        for r, c in removals:
            if r < self.rows and c < self.cols:
                # Shift all mappings after the removed position
                for j in range(c, self.cols-1):
                    self.mapping[r, j] = self.mapping[r, j+1]
    
    def get_pos_in_original(self, r, c):
        if r < self.rows and c < self.mapping.shape[1]:
            return self.mapping[r, c]
        return c  # Fallback to using the same position if out of bounds
        
