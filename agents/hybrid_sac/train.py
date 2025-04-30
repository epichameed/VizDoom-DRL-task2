import os
import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import wandb
from tqdm import tqdm

from ..common.utils import VizDoomWrapper, evaluate_agent
from .model import HybridSACNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Initialize networks
        self.model = HybridSACNetwork(
            env.action_space.n,
            frame_stack=config.frame_stack,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            hidden_dim=config.hidden_dim
        ).to(config.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.model.policy.parameters(), lr=config.learning_rate)
        self.q_optimizer = optim.Adam(list(self.model.qf1.parameters()) + 
                                    list(self.model.qf2.parameters()), lr=config.learning_rate)
        self.encoder_optimizer = optim.Adam(self.model.encoder.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(config.replay_capacity)
        
        # Initialize temperature parameter alpha
        self.target_entropy = -np.prod(env.action_space.n)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            features = self.model.encode(state)
            if evaluate:
                _, _, action = self.model.policy.sample(features)
            else:
                action, _, _ = self.model.policy.sample(features)
            return action.cpu().numpy()[0]
            
    def update_parameters(self):
        if len(self.memory) < self.config.batch_size:
            return
            
        # Sample from replay buffer
        transitions = self.memory.sample(self.config.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.config.device)
        action_batch = torch.FloatTensor(np.array(batch[1])).to(self.config.device)
        reward_batch = torch.FloatTensor(np.array(batch[2])).to(self.config.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.config.device)
        done_batch = torch.FloatTensor(np.array(batch[4])).to(self.config.device).unsqueeze(1)
        
        with torch.no_grad():
            # Get features for next states
            next_state_features = self.model.encode(next_state_batch)
            
            # Sample actions from policy
            next_state_action, next_state_log_pi, _ = self.model.policy.sample(next_state_features)
            
            # Compute Q-values for next states
            qf1_next_target, qf2_next_target = self.model.q_forward(next_state_features, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            # Compute target Q-values
            alpha = self.log_alpha.exp()
            next_q_value = min_qf_next_target - alpha * next_state_log_pi
            target_q_value = reward_batch + (1 - done_batch) * self.config.gamma * next_q_value
            
        # Get current Q-values
        current_features = self.model.encode(state_batch)
        qf1, qf2 = self.model.q_forward(current_features, action_batch)
        
        # Compute Q-function loss
        qf1_loss = F.mse_loss(qf1, target_q_value)
        qf2_loss = F.mse_loss(qf2, target_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Update Q-functions and encoder
        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        self.encoder_optimizer.step()
        
        # Compute policy loss
        pi, log_pi, _ = self.model.policy.sample(current_features.detach())
        qf1_pi, qf2_pi = self.model.q_forward(current_features.detach(), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((alpha * log_pi) - min_qf_pi).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return {
            'q1_loss': qf1_loss.item(),
            'q2_loss': qf2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
        
    def train(self):
        # Initialize wandb
        wandb.init(project="vizdoom-drl", config=self.config)
        
        for episode in tqdm(range(self.config.num_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for t in range(self.config.max_steps):
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.memory.push(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                episode_reward += reward
                self.steps_done += 1
                
                # Update parameters
                if self.steps_done % self.config.update_frequency == 0:
                    losses = self.update_parameters()
                    if losses is not None:
                        wandb.log(losses, step=self.steps_done)
                
                if done:
                    break
                    
            # Log episode results
            self.episode_rewards.append(episode_reward)
            wandb.log({
                "episode_reward": episode_reward,
                "steps": self.steps_done
            }, step=self.steps_done)
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                    'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                    'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
                    'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                    'steps_done': self.steps_done,
                    'episode_rewards': self.episode_rewards
                }, f"checkpoints/hybrid_sac_episode_{episode}.pt")
                
        wandb.finish()

def main():
    # Configuration
    class Config:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.scenario_path = "scenarios/basic.cfg"
            self.frame_stack = 4
            self.frame_skip = 4
            self.resolution = (84, 84)
            
            # Model parameters
            self.d_model = 512
            self.nhead = 8
            self.num_layers = 6
            self.dim_feedforward = 2048
            self.dropout = 0.1
            self.hidden_dim = 256
            
            # Training parameters
            self.num_episodes = 1000
            self.max_steps = 1000
            self.batch_size = 64
            self.learning_rate = 3e-4
            self.gamma = 0.99
            self.replay_capacity = 100000
            self.update_frequency = 4
            self.save_frequency = 100
            
    # Create environment
    config = Config()
    env = VizDoomWrapper(
        config.scenario_path,
        frame_skip=config.frame_skip,
        frame_stack=config.frame_stack,
        resolution=config.resolution
    )
    
    # Create agent and train
    agent = SACAgent(env, config)
    agent.train()
    
    # Evaluate final model
    eval_results = evaluate_agent(env, agent, n_episodes=10)
    print("Final evaluation results:", eval_results)
    
    env.close()

if __name__ == "__main__":
    main() 