import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class VisionTransformer(nn.Module):
    def __init__(self, frame_stack=4, resolution=(84, 84), patch_size=14, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        # Patch embedding
        self.patch_size = patch_size
        self.num_patches = (resolution[0] // patch_size) * (resolution[1] // patch_size)
        self.patch_embedding = nn.Conv2d(frame_stack, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def forward(self, x):
        # Handle input shape
        if len(x.shape) == 5:  # [batch_size, 1, frame_stack, H, W]
            x = x.squeeze(1)  # Remove the extra dimension
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch_size, d_model, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, d_model]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Use classification token for output
        return x[:, 0]

class ActorCritic(nn.Module):
    def __init__(self, n_actions, frame_stack=4, resolution=(84, 84), d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(ActorCritic, self).__init__()
        
        # Vision Transformer
        self.vision_transformer = VisionTransformer(
            frame_stack=frame_stack,
            resolution=resolution,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        features = self.vision_transformer(x)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
        
    def get_action(self, x):
        with torch.no_grad():
            action_probs, value = self.forward(x)
            action = torch.multinomial(action_probs, 1)
            return action.item(), action_probs[0, action.item()].item(), value.item() 