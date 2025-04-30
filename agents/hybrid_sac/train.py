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
from .model import HybridSAC

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.Transition(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = self.Transition(*zip(*transitions))
        return batch
        
    def __len__(self):
        return len(self.buffer)

class HybridSACAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = HybridSAC(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            frame_stack=config.frame_stack,
            resolution=config.resolution
        ).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        
        # Initialize memory
        self.memory = ReplayBuffer(config.memory_size)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if len(state.shape) == 4:  # Add batch dimension if missing
                state = state.unsqueeze(0)
                
            action_probs = self.actor(state)
            if evaluate:
                action = torch.argmax(action_probs, dim=1)
            else:
                action = torch.multinomial(action_probs, 1)
            return action.item()
            
    def predict(self, state, deterministic=True):
        """Predict action for evaluation"""
        return self.select_action(state, evaluate=deterministic), None
        
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
        done_batch = torch.cat(batch.done).to(self.device)
        
        # Compute actor loss
        action_probs = self.actor(state_batch)
        action_probs = action_probs.gather(1, action_batch)
        actor_loss = -torch.log(action_probs + 1e-8).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        
        return actor_loss.item()
        
    def train(self):
        # Initialize wandb
        wandb.init(project="vizdoom-drl", config=self.config)
        
        for episode in tqdm(range(self.config.num_episodes)):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            episode_reward = 0
            
            for t in range(self.config.max_steps):
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Store transition
                self.memory.push(
                    state,
                    action,
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
                
                if done:
                    break
                    
            # Log episode results
            self.episode_rewards.append(episode_reward)
            wandb.log({
                "episode_reward": episode_reward,
                "loss": loss if 'loss' in locals() else 0,
                "steps": self.steps_done
            }, step=self.steps_done)
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
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
            
            # Training parameters
            self.num_episodes = 1000
            self.max_steps = 1000
            self.batch_size = 32
            self.memory_size = 100000
            self.actor_lr = 1e-4
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
    agent = HybridSACAgent(env, config)
    agent.train()
    
    # Evaluate final model
    eval_results = evaluate_agent(env, agent, n_episodes=10)
    print("Final evaluation results:", eval_results)
    
    env.close()

if __name__ == "__main__":
    main() 