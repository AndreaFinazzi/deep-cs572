import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveDuelingQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(RecursiveDuelingQNet, self).__init__()
        # Recursive network for states encoding
        self.r_fc1 = nn.Linear(2*state_size, 128)
        # self.r_drop1 = nn.Dropout(p=0.5)
        # self.r_fc2 = nn.Linear(32, 64)
        # self.r_drop2 = nn.Dropout(p=0.5)
        self.r_fc3 = nn.Linear(128, state_size)

        self.fc1 = nn.Linear(state_size, 64)

        # State-Value Net
        self.state_fc1 = nn.Linear(64, 128)
        self.state_fc3 = nn.Linear(128, 1)

        # Action Net
        self.action_fc1 = nn.Linear(64, 128)
        self.action_fc3 = nn.Linear(128, action_size)
    
    # x = [t, t-1, t-2, t-3]
    def forward(self, x):
        # if x[3] :
        t2 = torch.cat([x[:, 3, :], x[:, 2, :]], dim=1)
        r = F.relu(self.r_fc1(t2))
        # r = self.r_drop1(r)
        # r = F.relu(self.r_fc2(r))
        # r = self.r_drop2(r)
        r_out = F.relu(self.r_fc3(r))
        # else:
        #     r_out = x[2]

        # if x[2] is not None:
        t3 = torch.cat([r_out, x[:, 1, :]], dim=1)
        r = F.relu(self.r_fc1(t3))
        # r = self.r_drop1(r)
        # r = F.relu(self.r_fc2(r))
        # r = self.r_drop2(r)
        r_out = F.relu(self.r_fc3(r))
        # else:
            # r_out = x[1]

        # if x[1] is not None:
        t4 = torch.cat([r_out, x[:, 0, :]], dim=1)
        r = F.relu(self.r_fc1(t4))
        # r = self.r_drop1(r)
        # r = F.relu(self.r_fc2(r))
        # r = self.r_drop2(r)
        r_out = F.relu(self.r_fc3(r))
        # else:
            # r_out = x[0]

        # Dueling with result of recursive net
        x_out = F.relu(self.fc1(r_out))

        state_value = F.relu(self.state_fc1(x_out))
        state_value = self.state_fc3(state_value)

        action_advantage = F.relu(self.action_fc1(x_out))
        action_advantage = self.action_fc3(action_advantage)

        action_mean = torch.mean(action_advantage, dim=1, keepdim=True)

        q_out = state_value + action_advantage - action_mean
        return q_out