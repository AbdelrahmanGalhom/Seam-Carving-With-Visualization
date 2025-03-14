import numpy as np
import cv2
from .DynamicMapping import DynamicMapping

class SeamCarving:
    def __init__(self,filename,out_height,out_width):
        self.filename = filename
        self.out_height = out_height
        self.out_width = out_width

        self.img = cv2.imread(filename).astype(np.float32) # reading image as float32 (no need for higher precision)
        self.in_height = self.img.shape[0] # height of the Input image
        self.in_width = self.img.shape[1] # width of the input image
        self.out_img = np.copy(self.img) # output image
        self.vis_img = np.copy(self.img) # visualization image

        self.mapping = DynamicMapping(self.in_height, self.in_width)

        self.seam_carving()

    def _seam_carving(self):
        rows_to_remove = self.in_height - self.out_height
        cols_to_remove = self.in_width - self.out_width

        # Remove columns if needed (reduce width)
        if cols_to_remove > 0:
            self.seams_removal(cols_to_remove,True)
        
        # Remove rows if needed (reduce height)
        if rows_to_remove > 0:
            self.out_img = self.rotate_image(self.out_img, 1)
            self.seams_removal(rows_to_remove,False)
            self.out_img = self.rotate_image(self.out_img, 0)
    

    def _seams_removal(self, pixels, cols=True):
        for i in range(pixels):
            energy_map = self.calc_energy_map()
            seam_idx = self.dp_cumulative_map(energy_map)
            
            if cols:
                self.vis_img = self.visualize_seam(seam_idx)
            
            # Only pass valid positions within current image dimensions
            valid_removals = []
            for r in range(len(seam_idx)):
                if r < self.out_img.shape[0] and seam_idx[r] < self.out_img.shape[1]:
                    valid_removals.append((r, seam_idx[r]))
                    
            self.mapping.apply_removals(valid_removals)
            self.delete_seam(seam_idx)


    def _visualize_seam(self, seam_idx):
        m, n = self.vis_img.shape[: 2]
        output = np.copy(self.vis_img)
        
        # Get the seam pixels mapped to the original image
        for row in range(min(m, len(seam_idx))):
            if seam_idx[row] < self.out_img.shape[1]:
                col = self.mapping.get_pos_in_original(row, seam_idx[row])
                if 0 <= col < n:  # Ensure index is in bounds
                    output[row, col] = [0, 0, 255]
        
        return output
    
    def _calc_energy_map(self):
        b, g, r = cv2.split(self.out_img) # splitting the image into its 3 channels

        #There are multiple ways to calculate the energy map of the image
        #1. Summing the absolute values of the gradients in the x and y directions
        # b_energy = np.absolute(cv2.Sobel(b, -1, 1, 0)) + np.absolute(cv2.Sobel(b, -1, 0, 1)) 
        # g_energy = np.absolute(cv2.Sobel(g, -1, 1, 0)) + np.absolute(cv2.Sobel(g, -1, 0, 1))
        # r_energy = np.absolute(cv2.Sobel(r, -1, 1, 0)) + np.absolute(cv2.Sobel(r, -1, 0, 1))

        #2. sqrt((dI/dx)^2 + (dI/dy)^2)
        #b_energy = np.sqrt(cv2.Sobel(b, -1, 1, 0)**2 + cv2.Sobel(b, -1, 0, 1)**2)
        #g_energy = np.sqrt(cv2.Sobel(g, -1, 1, 0)**2 + cv2.Sobel(g, -1, 0, 1)**2)
        #r_energy = np.sqrt(cv2.Sobel(r, -1, 1, 0)**2 + cv2.Sobel(r, -1, 0, 1)**2)

        #3. Using roll to calculate the gradient
        b_energy = np.abs(np.roll(b, -1, axis=1) - np.roll(b, 1, axis=1)) + np.abs(np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0))
        g_energy = np.abs(np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) + np.abs(np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0))
        r_energy = np.abs(np.roll(r, -1, axis=1) - np.roll(r, 1, axis=1)) + np.abs(np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))

        return b_energy + g_energy + r_energy
    
    def _calc_energy_map_grayscale(self):
        # Convert to uint8 format for cvtColor operation
        img_for_gray = np.clip(self.out_img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_for_gray, cv2.COLOR_BGR2GRAY)
        # Convert back to float for calculations
        gray = gray.astype(np.float32)
        # Calculate energy by roll method
        energy = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1)) + np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
        return energy
    
    
    def _dp_cumulative_map(self,energy_map):
        h, w = energy_map.shape
        dp = energy_map.copy()
        backtrack = np.zeros_like(dp, dtype=np.int32)
        
        for i in range(1, h):
            for j in range(w):
                min_col = j
                if j > 0 and dp[i-1, j-1] < dp[i-1, min_col]:
                    min_col = j - 1
                if j < w-1 and dp[i-1, j+1] < dp[i-1, min_col]:
                    min_col = j + 1
                dp[i, j] += dp[i-1, min_col]
                backtrack[i, j] = min_col
        
        seam = []
        min_idx = np.argmin(dp[-1])
        for i in range(h-1, -1, -1):
            seam.append(min_idx)
            min_idx = backtrack[i, min_idx]
        return seam[::-1]

    def _delete_seam(self, seam_idx):
        m, n = self.out_img.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_img[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_img[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_img[row, :, 2], [col])
        self.out_img = np.copy(output)

    def _rotate_image(self, image, ccw):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return output
    
    def save_output_image(self):
        img_name = self.filename.split(".")[0]
        cv2.imwrite(f"{img_name}_output.jpg", self.out_img)
    
    def save_visualization_image(self):
        img_name = self.filename.split(".")[0]
        cv2.imwrite(f"{img_name}_visualization.jpg", self.vis_img)


