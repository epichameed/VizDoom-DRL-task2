import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, frame_stack=4, resolution=(84, 84)):
        super(DQN, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
            
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(resolution[0], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(resolution[1], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Q-value head
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
        # Handle input shape
        if len(x.shape) == 5:  # [batch_size, 1, frame_stack, H, W]
            x = x.squeeze(1)  # Remove the extra dimension
            
        # Extract features
        features = self.cnn(x)
        
        # Get Q-values
        q_values = self.fc(features)
        
        return q_values

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