import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm

from ..common.utils import VizDoomWrapper, evaluate_agent
from .model import ActorCritic

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batch(self):
        # Stack tensors along the first dimension
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        probs = torch.tensor(self.probs, dtype=torch.float)
        vals = torch.tensor(self.vals, dtype=torch.float)
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        dones = torch.tensor(self.dones, dtype=torch.float)
        
        return states, actions, probs, vals, rewards, dones
               
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PPOAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor-critic network
        self.policy = ActorCritic(
            env.action_space.n,
            frame_stack=config.frame_stack,
            resolution=config.resolution,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        ).to(config.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Initialize memory
        self.memory = PPOMemory()
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if len(state.shape) == 4:  # Add batch dimension if missing
                state = state.unsqueeze(0)
                
            action_probs, value = self.policy(state)
            
            # Add validation to ensure valid probability distribution
            # Check for NaN or Inf values
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                # Replace with uniform distribution as fallback
                action_probs = torch.ones_like(action_probs) / action_probs.shape[1]
            
            # Ensure all values are positive
            action_probs = torch.clamp(action_probs, min=1e-6)
            
            # Re-normalize to ensure sum to 1
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
            if evaluate:
                action = torch.argmax(action_probs, dim=1)
            else:
                action = torch.multinomial(action_probs, 1)
            
            return action.item(), action_probs[0, action.item()].item(), value.item()
            
    def predict(self, state, deterministic=True):
        """Predict action for evaluation"""
        return self.select_action(state, evaluate=deterministic)[0], None
        
    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        gae = 0
        returns = []
        next_value = 0  # Initialize next_value for the last step
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0  # No next value for the last step
            else:
                next_value = values[step + 1]
                
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * lambda_ * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns
        
    def train(self):
        # Initialize wandb
        wandb.init(project="vizdoom-drl", config=self.config)
        
        for episode in tqdm(range(self.config.num_episodes)):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            episode_reward = 0
            
            for t in range(self.config.max_steps):
                # Select action
                action, prob, val = self.select_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Store transition
                self.memory.states.append(state.cpu())
                self.memory.actions.append(action)
                self.memory.probs.append(prob)
                self.memory.vals.append(val)
                self.memory.rewards.append(reward)
                self.memory.dones.append(done)
                
                # Move to next state
                state = next_state
                episode_reward += reward
                self.steps_done += 1
                
                if done:
                    break
                    
            # Log episode results
            self.episode_rewards.append(episode_reward)
            wandb.log({
                "episode_reward": episode_reward,
                "steps": self.steps_done
            }, step=self.steps_done)
            
            # Update policy
            if len(self.memory.states) >= self.config.batch_size:
                self.update_policy()
                
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'steps_done': self.steps_done,
                    'episode_rewards': self.episode_rewards
                }, f"checkpoints/transformer_ppo_episode_{episode}.pt")
                
        wandb.finish()
        
    def update_policy(self):
        # Generate batch
        states, actions, old_probs, vals, rewards, dones = self.memory.generate_batch()
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        old_probs = old_probs.to(self.config.device)
        vals = vals.to(self.config.device)
        rewards = rewards.to(self.config.device)
        dones = dones.to(self.config.device)
        
        # Compute returns and advantages
        returns = self.compute_gae(rewards, vals, dones)
        returns = torch.tensor(returns).to(self.config.device)
        
        # Ensure tensors have the same size
        if len(returns) > len(vals):
            returns = returns[:-1]  # Remove the last element to match vals size
            
        advantages = returns - vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for n epochs
        for _ in range(self.config.n_epochs):
            # Get new action probabilities and values
            new_probs, new_vals = self.policy(states)
            
            # Add validation to ensure valid probability distribution
            if torch.isnan(new_probs).any() or torch.isinf(new_probs).any():
                # Skip this update iteration if probabilities are invalid
                continue
                
            # Ensure all values are positive and normalized
            new_probs = torch.clamp(new_probs, min=1e-6)
            new_probs = new_probs / new_probs.sum(dim=1, keepdim=True)
            
            # Get probabilities for the actions that were taken
            action_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute ratios with numerical stability
            # Add small epsilon to prevent division by zero or log of zero
            ratios = torch.exp(torch.log(action_probs + 1e-10) - torch.log(old_probs + 1e-10))
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(new_vals.squeeze(), returns)
            
            # Compute entropy loss with numerical stability
            entropy_loss = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # Log losses
            wandb.log({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "total_loss": loss.item()
            }, step=self.steps_done)
            
        # Clear memory
        self.memory.clear_memory()

def main():
    # Configuration
    class Config:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.scenario_path = "scenarios/basic.cfg"
            self.frame_stack = 4
            self.frame_skip = 4
            self.resolution = (84, 84)
            
            # Transformer parameters
            self.d_model = 512
            self.nhead = 8
            self.num_layers = 6
            self.dim_feedforward = 2048
            self.dropout = 0.1
            
            # Training parameters
            self.num_episodes = 1000
            self.max_steps = 1000
            self.batch_size = 64
            self.learning_rate = 3e-4
            self.gamma = 0.99
            self.lambda_ = 0.95
            self.clip_epsilon = 0.2
            self.n_epochs = 10
            self.value_coef = 0.5
            self.entropy_coef = 0.01
            self.max_grad_norm = 0.5
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
    agent = PPOAgent(env, config)
    agent.train()
    
    # Evaluate final model
    eval_results = evaluate_agent(env, agent, n_episodes=10)
    print("Final evaluation results:", eval_results)
    
    env.close()

if __name__ == "__main__":
    main()