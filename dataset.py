from torch.utils.data import Dataset
import os
import numpy as np
import torch
import cv2

"""
The dataset class downsamples the images by a factor of 4 like in the MCNN paper
"""

class Dataset(Dataset):
    
    def __init__(self, image_dir, density_dir):

        self.image_dir = image_dir 
        self.density_dir = density_dir
        self.downsample = 4 # Images need to be downsampled by 4 for tensors
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.jpg')]
        self.samples = len(self.images)


    # PyTorch default method
    def __len__(self):
        return self.samples  # Needed to calculate evaluation metrics


    # PyTorch default method
    def __getitem__(self, i):
        
        # Create pair with an image and its corresponding density map 
        img_title = self.images[i]
        img_join = os.path.join(self.image_dir, img_title)
        img = cv2.imread(img_join)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dmap_path = os.path.join(self.density_dir, img_title.replace('.jpg', '.npy'))
        gt_dmap = np.load(dmap_path)

        # Downsample the image and density map pair and convert to tensors "https://learnopencv.com/image-resizing-with-opencv/"
        h, w, c = img.shape # get height and width of the image (we don't need the colour channel)
        new_h = h - (h % self.downsample)
        new_w = w - (w % self.downsample)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC) # Images use cubic to retain better image quality, this process is slower than linear "https://medium.com/@nasuhcanturker/what-is-interpolation-ceb7b489943e"

        dmap_h = new_h // self.downsample
        dmap_w = new_w // self.downsample
        gt_dmap = cv2.resize(gt_dmap, (dmap_w, dmap_h), interpolation=cv2.INTER_LINEAR) # density map uses linear interpolation to better reatain head count "https://medium.com/@nasuhcanturker/what-is-interpolation-ceb7b489943e"
        gt_dmap = gt_dmap * (self.downsample ** 2)  # adjust density values so the head count remains consistent after downsampling

        # Convert to tensor
        img = img.transpose(2, 0, 1)  # to CxHxW
        img_tensor = torch.tensor(img, dtype=torch.float32)
        gt_dmap_tensor = torch.tensor(gt_dmap[np.newaxis, :, :], dtype=torch.float32)

        return img_tensor, gt_dmap_tensor
