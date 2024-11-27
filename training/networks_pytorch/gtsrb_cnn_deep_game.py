# From https://github.com/TrustAI/DeepGame/blob/master/NeuralNetwork.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GTSRB_CNN_Deep_Game(torch.nn.Module):

    def __init__(self):
        num_classes = 43

        super(GTSRB_CNN_Deep_Game, self).__init__()
        self.name = "gtsrb_cnn_deep_game"
         # Define the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = x.view(-1, 128 * 8 * 8)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x