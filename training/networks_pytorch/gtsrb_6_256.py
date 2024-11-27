import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRB_6_256(nn.Module):
    def __init__(self):
        super(GTSRB_6_256, self).__init__()
        self.name = "gtsrb_6_256"
        self.layer1 = nn.Linear(3*32*32, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 43)


    def forward(self,x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
        return x
    


# From https://arxiv.org/pdf/1912.09533