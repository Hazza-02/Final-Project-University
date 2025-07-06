import torch
import torch.nn as nn

"""

--- BASE MODEL LOGIC ---

Model archtiecture from "https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf"

The model features three receptive fields (or columns), large, medium, and small to handle details and different distances,
Kernel size and padding decrease the smaller the receptive field 
The model uses the ReLU (Rectified Linear Unit) activation function and maxpooling for intermediate layers

The weight initilisation was taking from an existing solution see "https://github.com/CommissarMa/MCNN-pytorch/blob/master/mcnn_model.py"
See System design section of dissertation for reasoning of this 

"""
                                          
class MCNN(nn.Module):

    def __init__(self,load_weights=False):
        super(MCNN,self).__init__()             
        
        # Large receptive field         
        self.branch1 = nn.Sequential(   
            nn.Conv2d(3, 16, kernel_size=9, padding=4),  
            nn.ReLU(),                  
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.ReLU()
        )
        
        # Medium receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Small receptive field
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fuse = nn.Sequential(nn.Conv2d(30, 1, kernel_size=1, padding=0)) # Connect layers
        
        # Intilisation for weights from https://github.com/CommissarMa/MCNN-pytorch/blob/master/mcnn_model.py , Lines 69 to 80 is not my code
        if not load_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                                  
    def forward(self, x):
        
        # Process image (x) through all columns
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x1, x2, x3), dim=1) # Concatenate the outputs from all branches
        x = self.fuse(x) # Pass through the final convolution layer
        
        return x
