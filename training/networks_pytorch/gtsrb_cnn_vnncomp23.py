import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRB_CNN_VNNCOMP23(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "gtsrb_cnn_vnncomp23"
        self.conv1 = nn.Conv2d(3,32, kernel_size=5, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size=5, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(576, 1024)  # Adjust according to your input size
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 43) 
    def forward(self,x):
        x = self.conv1_bn(F.max_pool2d(F.relu(self.conv1(x)),2))
        x = self.conv2_bn(F.max_pool2d(F.relu(self.conv2(x)),2))
        x = self.conv3_bn(F.max_pool2d(F.relu(self.conv3(x)),2))
        x = torch.flatten(x, 1)
  
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# From https://arxiv.org/pdf/2310.03033