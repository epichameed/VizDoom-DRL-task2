import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import wandb
from tqdm import tqdm
import math

from ..common.utils import VizDoomWrapper, evaluate_agent
from .model import DQN, DuelingDQN

# Define the transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            frame_stack=config.frame_stack,
            resolution=config.resolution
        ).to(self.device)
        self.target_net = DQN(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            frame_stack=config.frame_stack,
            resolution=config.resolution
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        # Initialize memory
        self.memory = ReplayBuffer(config.memory_size)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
            
    def predict(self, state, deterministic=True):
        """Predict action for evaluation"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if len(state.shape) == 4:  # Add batch dimension if missing
                state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            if deterministic:
                action = q_values.max(1)[1].item()
            else:
                action = torch.multinomial(F.softmax(q_values, dim=1), 1).item()
            return action, None  # Return action and None for compatibility with other agents
        
    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return
            
        # Sample from memory
        batch = self.memory.sample(self.config.batch_size)
        
        # Convert to tensors
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device).float()  # Convert to float
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
            expected_state_action_values = (next_state_values * (1 - done_batch) * self.config.gamma) + reward_batch
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
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
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            episode_reward = 0
            
            for t in range(self.config.max_steps):
                # Select action
                epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                         math.exp(-1. * self.steps_done / self.config.epsilon_decay)
                action = self.select_action(state, epsilon)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Store transition
                self.memory.push(
                    state,
                    action.item(),
                    torch.tensor([reward], device=self.device),
                    next_state,
                    torch.tensor([done], device=self.device)
                )
                
                # Move to next state
                state = next_state
                episode_reward += reward
                self.steps_done += 1
                
                # Optimize model
                loss = self.optimize_model()
                
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
                "loss": loss if 'loss' in locals() else 0,
                "steps": self.steps_done
            }, step=self.steps_done)
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                torch.save({
                    'policy_state_dict': self.policy_net.state_dict(),
                    'target_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'steps_done': self.steps_done,
                    'episode_rewards': self.episode_rewards
                }, f"checkpoints/dqn_episode_{episode}.pt")
                
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
            self.memory_size = 100000
            self.learning_rate = 1e-4
            self.gamma = 0.99
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.epsilon_decay = 10000
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