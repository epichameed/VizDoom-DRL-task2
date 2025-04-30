import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions, frame_stack=4):
        super(DQN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        # Forward pass through CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DuelingDQN(nn.Module):
    def __init__(self, n_actions, frame_stack=4):
        super(DuelingDQN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Value stream
        self.value_fc1 = nn.Linear(64 * 7 * 7, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(64 * 7 * 7, 512)
        self.advantage_fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        # Forward pass through CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values 