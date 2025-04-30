import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNEncoder(nn.Module):
    def __init__(self, frame_stack=4):
        super(CNNEncoder, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class HybridEncoder(nn.Module):
    def __init__(self, frame_stack=4, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(HybridEncoder, self).__init__()
        
        # CNN encoder
        self.cnn = CNNEncoder(frame_stack)
        
        # Project CNN features to transformer dimension
        self.projection = nn.Linear(64 * 7 * 7, d_model)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
    def forward(self, x):
        # CNN encoding
        x = self.cnn(x)
        batch_size = x.size(0)
        
        # Reshape for transformer
        x = x.view(batch_size, -1)
        x = self.projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Transformer encoding
        x = self.transformer(x)
        return x.squeeze(1)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        
        # Enforcing action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
        
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

class HybridSACNetwork(nn.Module):
    def __init__(self, n_actions, frame_stack=4, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, hidden_dim=256):
        super(HybridSACNetwork, self).__init__()
        
        # Feature encoder
        self.encoder = HybridEncoder(
            frame_stack=frame_stack,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Policy network
        self.policy = GaussianPolicy(d_model, n_actions, hidden_dim)
        
        # Q-networks
        self.qf1 = QNetwork(d_model, n_actions, hidden_dim)
        self.qf2 = QNetwork(d_model, n_actions, hidden_dim)
        
    def encode(self, x):
        return self.encoder(x)
        
    def policy_forward(self, state):
        return self.policy(state)
        
    def q_forward(self, state, action):
        return self.qf1(state, action), self.qf2(state, action)
        
    def q1_forward(self, state, action):
        return self.qf1.q1_forward(state, action)

class HybridSAC(nn.Module):
    def __init__(self, state_dim, n_actions, frame_stack=4, resolution=(84, 84)):
        super(HybridSAC, self).__init__()
        
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
        
        # Policy head
        self.policy = nn.Sequential(
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
        
        # Get action probabilities
        action_logits = self.policy(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs 