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

        # kernels for calculating the gradient of the image(used for energy calculation)
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        self.seam_carving()

    def seam_carving(self):
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
    

    def seams_removal(self, pixels, cols=True):
        for i in range(pixels):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            
            if cols:
                self.vis_img = self.visualize_seam(seam_idx)
            
            # Only pass valid positions within current image dimensions
            valid_removals = []
            for r in range(len(seam_idx)):
                if r < self.out_img.shape[0] and seam_idx[r] < self.out_img.shape[1]:
                    valid_removals.append((r, seam_idx[r]))
                    
            self.mapping.apply_removals(valid_removals)
            self.delete_seam(seam_idx)


    def visualize_seam(self, seam_idx):
        m, n = self.vis_img.shape[: 2]
        output = np.copy(self.vis_img)
        
        # Get the seam pixels mapped to the original image
        for row in range(min(m, len(seam_idx))):
            if seam_idx[row] < self.out_img.shape[1]:
                col = self.mapping.get_pos_in_original(row, seam_idx[row])
                if 0 <= col < n:  # Ensure index is in bounds
                    output[row, col] = [0, 0, 255]
        
        return output
    
    def calc_energy_map(self):
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
    
    def calc_energy_map_grayscale(self):
        # Convert to uint8 format for cvtColor operation
        img_for_gray = np.clip(self.out_img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_for_gray, cv2.COLOR_BGR2GRAY)
        # Convert back to float for calculations
        gray = gray.astype(np.float32)
        # Calculate energy by roll method
        energy = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1)) + np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
        return energy
    
    
    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m): # starting from the second row
            for col in range(n): # for each pixel in the row
                #The boundary checks max(col-1, 0) and min(col+2, n-1) ensure the algorithm doesn't attempt to access pixels outside the image edges
                #The slice notation output[row-1, max(col-1, 0): min(col+2, n-1)] selects a small window of 1-3 pixels from the previous row, and np.amin() finds the minimum value in this range.
                output[row, col] = energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return output
    
    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output
    
    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_img)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output
    
    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32) #initializes an array (output) of zeros with a length equal to the image height, which will store the horizontal positions (column indices) of the seam pixels. 
        output[-1] = np.argmin(cumulative_map[-1]) # the pixel with minimum energy in the bottom row is the starting point of the seam
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0: # if the seam is on the left edge of the image
                output[row] = np.argmin(cumulative_map[row, : 2])
            else: 
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output

    def delete_seam(self, seam_idx):
        m, n = self.out_img.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_img[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_img[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_img[row, :, 2], [col])
        self.out_img = np.copy(output)

    def rotate_image(self, image, ccw):
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
        cv2.imwrite(f"{self.filename}_output.jpg", self.out_img)
    
    def save_visualization_image(self):
        cv2.imwrite(f"{self.filename}_visualization.jpg", self.vis_img)

