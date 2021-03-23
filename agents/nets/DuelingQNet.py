import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNet, self).__init__()

        # Input processing
        # self.fc1 = nn.Linear(state_size, 64)
        # self.drop1 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(64, 128)
        # self.drop2 = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(128, 256)

        # # State-Value Net
        # self.state_fc1 = nn.Linear(256, 128)
        # self.state_fc2 = nn.Linear(128, 64)
        # self.state_fc3 = nn.Linear(64, 1)

        # # Action Net
        # self.action_fc1 = nn.Linear(256, 128)
        # self.action_fc2 = nn.Linear(128, 128)
        # self.action_fc3 = nn.Linear(128, action_size)

        self.fc1 = nn.Linear(state_size, 64)
        # self.drop1 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(64, 128)
        # self.drop2 = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(128, 256)

        # State-Value Net
        self.state_fc1 = nn.Linear(64, 256)
        self.state_fc3 = nn.Linear(256, 1)

        # Action Net
        self.action_fc1 = nn.Linear(64, 256)
        self.action_fc3 = nn.Linear(256, action_size)
        
    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = self.drop1(x)
        # x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        # x_out = F.relu(self.fc3(x))

        # state_value = F.relu(self.state_fc1(x_out))
        # state_value = F.relu(self.state_fc2(state_value))
        # state_value = self.state_fc3(state_value)

        # action_advantage = F.relu(self.action_fc1(x_out))
        # action_advantage = F.relu(self.action_fc2(action_advantage))
        # action_advantage = self.action_fc3(action_advantage)

        x_out = F.relu(self.fc1(x))

        state_value = F.relu(self.state_fc1(x_out))
        state_value = self.state_fc3(state_value)

        action_advantage = F.relu(self.action_fc1(x_out))
        action_advantage = self.action_fc3(action_advantage)

        action_mean = torch.mean(action_advantage, dim=1, keepdim=True)

        q_out = state_value + action_advantage - action_mean
        return q_out