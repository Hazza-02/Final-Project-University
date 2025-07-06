import torch
import torch.nn as nn
import torch.nn.functional as F

"""

--- BASE MODEL LOGIC ---

Model archtiecture from "https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf"

The model features three receptive fields (or columns), large, medium, and small to handle details and different distances,
Kernel size and padding decrease the smaller the receptive field 
The model uses the ReLU (Rectified Linear Unit) activation function and maxpooling for intermediate layers

The weight initilisation was taking from an existing solution see "https://github.com/CommissarMa/MCNN-pytorch/blob/master/mcnn_model.py"
See System design section of dissertation for reasoning of this 

--- CBAM MODIFICATION ---

CBAM is added after each column in the MCNN architecture 
basis of CBAM implementation from "https://github.com/nikhilroxtomar/Attention-Mechanism-Implementation/blob/main/PyTorch/cbam.py"

"""

# For CBAM, Channel_Attention, Spatial_Attention all taken from "https://github.com/nikhilroxtomar/Attention-Mechanism-Implementation/blob/main/PyTorch/cbam.py"
# Channel attention 
class Channel_Attention(nn.Module):
    def __init__(self, channel, ratio = 8):
            super().__init__()
            
            self.average_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            
            self.mlp = nn.Sequential (
                nn.Linear(channel, channel // ratio, bias = False),
                nn.ReLU(inplace = True),
                nn.Linear(channel // ratio, channel, bias = False)
            )
            self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        
        x1 = self.average_pool(x).view(x.size(0), -1) # "https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-shapes-cannot-be-multiplied-64x13056-and-153600x2048/101315/6"
        x1 = self.mlp(x1)
    
        x2 = self.max_pool(x).view(x.size(0), -1) 
        x2 = self.mlp(x2)
        
        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        
        refined_feats = x * feats
        return refined_feats
       
# Spatial attention taken from "https://github.com/nikhilroxtomar/Attention-Mechanism-Implementation/blob/main/PyTorch/cbam.py"
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding = 3, bias = False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        x1 = torch.mean(x, dim = 1, keepdim = True)
        x2, _ = torch.max(x, dim = 1, keepdim = True)
        feats = torch.cat( [x1, x2], dim = 1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats
        
        return refined_feats
 
# CBAM module taken "https://github.com/nikhilroxtomar/Attention-Mechanism-Implementation/blob/main/PyTorch/cbam.py"
class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        
        self.ca = Channel_Attention(channel)
        self.sa = Spatial_Attention()
        
    def forward(self, x):
        
        x = self.ca(x)
        x = self.sa(x)
        return x
    
# MCNN model with CBAM
class MCNN_CBAM(nn.Module):
    def __init__(self, load_weights=False):
        super(MCNN_CBAM, self).__init__()
        
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
        self.attn1 = CBAM(8)

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
        self.attn2 = CBAM(10)

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
        self.attn3 = CBAM(12)
        
        self.fuse = nn.Sequential(nn.Conv2d(30, 1, kernel_size=1))  # Connect layers

        # Intilisation for weights from https://github.com/CommissarMa/MCNN-pytorch/blob/master/mcnn_model.py
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
        x1 = self.attn1(self.branch1(x))
        x2 = self.attn2(self.branch2(x))
        x3 = self.attn3(self.branch3(x))
        x = torch.cat((x1, x2, x3), dim=1)  # Concatenate the outputs from all branches
        x = self.fuse(x) # Pass through the final convolution layer
        return x
    
