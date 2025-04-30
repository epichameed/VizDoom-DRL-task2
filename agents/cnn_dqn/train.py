import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import wandb
from tqdm import tqdm

from ..common.utils import VizDoomWrapper, evaluate_agent
from .model import DQN, DuelingDQN

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Initialize networks
        self.policy_net = DuelingDQN(env.action_space.n, config.frame_stack).to(config.device)
        self.target_net = DuelingDQN(env.action_space.n, config.frame_stack).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(config.replay_capacity)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.config.device)
        
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
            
    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return
            
        # Sample from replay buffer
        transitions = self.memory.sample(self.config.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            next_state_values[done_batch] = 0.0
            expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch
            
        # Compute loss and optimize
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()
        
    def train(self):
        # Initialize wandb
        wandb.init(project="vizdoom-drl", config=self.config)
        
        for episode in tqdm(range(self.config.num_episodes)):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            episode_reward = 0
            
            for t in range(self.config.max_steps):
                # Select action
                epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                         np.exp(-1. * self.steps_done / self.config.epsilon_decay)
                action = self.select_action(state, epsilon)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.config.device)
                
                # Store transition
                self.memory.push(state, action, torch.tensor([reward], device=self.config.device),
                               next_state, torch.tensor([done], device=self.config.device))
                
                # Move to next state
                state = next_state
                episode_reward += reward
                self.steps_done += 1
                
                # Optimize model
                if self.steps_done % self.config.update_frequency == 0:
                    loss = self.optimize_model()
                    wandb.log({"loss": loss}, step=self.steps_done)
                
                # Update target network
                if self.steps_done % self.config.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if done:
                    break
                    
            # Log episode results
            self.episode_rewards.append(episode_reward)
            wandb.log({
                "episode_reward": episode_reward,
                "epsilon": epsilon,
                "steps": self.steps_done
            }, step=self.steps_done)
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                torch.save({
                    'policy_net_state_dict': self.policy_net.state_dict(),
                    'target_net_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'steps_done': self.steps_done,
                    'episode_rewards': self.episode_rewards
                }, f"checkpoints/cnn_dqn_episode_{episode}.pt")
                
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
            
            # Training parameters
            self.num_episodes = 1000
            self.max_steps = 1000
            self.batch_size = 32
            self.gamma = 0.99
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.epsilon_decay = 10000
            self.learning_rate = 1e-4
            self.replay_capacity = 100000
            self.update_frequency = 4
            self.target_update = 1000
            self.max_grad_norm = 1.0
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
    agent = DQNAgent(env, config)
    agent.train()
    
    # Evaluate final model
    eval_results = evaluate_agent(env, agent, n_episodes=10)
    print("Final evaluation results:", eval_results)
    
    env.close()

if __name__ == "__main__":
    main() 