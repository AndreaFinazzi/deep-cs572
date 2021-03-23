import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        out = self.fc3(x)
        return out