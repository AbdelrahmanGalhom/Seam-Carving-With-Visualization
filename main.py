import cv2
from src.SeamCarving import SeamCarving 

if __name__ == "__main__":
    img_name = "image.png" #insert the image name
    image = cv2.imread(img_name) #read the image
    print(image.shape)
    sc = SeamCarving(img_name, 350,200)
    print(sc.out_img.shape)
    sc.save_output_image()
    sc.save_visualization_image()
    print("Done")
