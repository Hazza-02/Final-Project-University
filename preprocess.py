import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from sklearn.neighbors import NearestNeighbors

"""

--- Preprocess Logic ---

CNN crowd counting models learn much better when using density maps (heat maps) as
the model has a better idea of what peoples heads are and what densities look like

 
--- Geometry-adaptive kernels ---

From - Single-Image Crowd Counting via Multi-Column Convolutional Neural Network
As found - https://ieeexplore.ieee.org/document/7780439

Why? - we could use a fixed Gaussian kernel on all heads to create a heat map.

However, this does not solve the issue of perspective and distance, if a head 
is further away then a larger fixed kernel is not going to accurately depict this.
So by using geometry-adaptive kernels the issue of distance is solved as the kernel 
will become smaller and adjust the further distance away a head is. In short the kernel
adapts the the geometry of the scene.

How does this work? - for a head we work out the k nearest neighboring heads,
then calculate the average distance between these heads which gives an idea at how far away
the head is and so we can use this to determine the size of the kernel.

"""

class Preprocess:
    
    def __init__(self, image_dir, gt_dir, output_dir):

        self.image_dir = image_dir # directory to the image
        self.gt_dir = gt_dir # directory to the ground truth file
        self.output_dir = output_dir # directory to the created density maps
    
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    # Loading mat files from https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
    def load_coordinates(self, mat_dir):

        mat_data = sio.loadmat(mat_dir)
        image_info = mat_data['image_info'][0, 0] 
        head_locations = image_info['location'][0, 0]
        head_coordinates = np.array(head_locations, dtype=np.float32)

        return head_coordinates
   
    # Using geometry-adaptive kernels
    # beta is 0.3 this is a fixed value determined to be optimal, found in MCNN paper, and K_nearest 2.
    def generate_density_map(self, head_positions, image_shape, k_nearest=2, beta=0.3):
        
        density_map = np.zeros(image_shape, dtype=np.float32) # create a blank density map
        head_positions = np.array(head_positions) # coordinates of heads
        
        # Find the average distance between the nearest heads for adaptive sigma
        # SciPy manual for Nearest neighbors https://scikit-learn.org/stable/modules/neighbors.html
        neighbour = NearestNeighbors(n_neighbors=k_nearest + 1, algorithm='auto').fit(head_positions)
        distances, _ = neighbour.kneighbors(head_positions)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        for i in range(len(head_positions)):
            
            x, y = head_positions[i]
            avg_distance = avg_distances[i]
            sigma = beta * avg_distance # calcualtes adaptive sigma for gaussian blur
            x = int(x)
            y = int(y)
            
            # Create a map for just one person so a gaussian filter cane be applied to them before adding to the density map as the gaussian filter works on the image as a whole
            single_person = np.zeros(image_shape, dtype=np.float32)
            single_person[y, x] = 1 # the 1 represents the person 
            single_person = nd.gaussian_filter(single_person, sigma) # SciPy manual for gaussian filter https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
            density_map += single_person
        
        return density_map
    
    def process_dataset(self):
  
        image_filenames = os.listdir(self.image_dir)
      
        for img_filename in image_filenames:
            
           # get direcotories and coordinates 
            img_path = os.path.join(self.image_dir, img_filename)
            gt_path = os.path.join(self.gt_dir, f"GT_{os.path.splitext(img_filename)[0]}.mat")
            gt_coordinates = self.load_coordinates(gt_path)
            img = cv2.imread(img_path)
            
            image_shape = (img.shape[0], img.shape[1])  # (height, width) of the image

            # create the density map density map
            density_map = self.generate_density_map(gt_coordinates, image_shape)

            # Check density map head count matches ground truth head count
            print(f"Image: {img_filename}")
            print(f"GT Count: {len(gt_coordinates)}, Density Map Sum: {np.sum(density_map):.2f}")

            # Save density map
            density_filename = os.path.splitext(img_filename)[0] + ".npy"
            np.save(os.path.join(self.output_dir, density_filename), density_map)
            
            
# Test code to visually see a density map
def show_density_map(path):
    density_map = np.load(path)
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.title('Density Map')
    plt.show()
    print(f"Total count in density map: {np.sum(density_map):.2f}")
       
if __name__ == '__main__':
    
    show_density_map(r"ShanghaiTech\part_A\test_data\density_maps\IMG_5.npy")     
    base_path = r"D:\Final Project (University)\ShanghaiTech\Part_A"
    image_dir = os.path.join(base_path, "test_data", "images") 
    gt_dir = os.path.join(base_path, "test_data", "ground-truth")
    output_dir = os.path.join(base_path, "test_data", "density_maps")
    preprocessor = Preprocess(image_dir, gt_dir, output_dir)
    preprocessor.process_dataset()