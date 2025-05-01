import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from agents.common.utils import VizDoomWrapper, evaluate_agent
from agents.cnn_dqn.model import DuelingDQN
from agents.transformer_ppo.model import ActorCritic
from agents.hybrid_sac.model import HybridSACNetwork
from agents.cnn_dqn.config import Config as DQNConfig
from agents.transformer_ppo.config import Config as PPOConfig
from agents.hybrid_sac.config import Config as SACConfig

def load_agent(agent_type, checkpoint_path, env):
    """Load a trained agent from checkpoint"""
    # Check if we're in Colab by looking at the file path
    is_colab = os.path.exists("/content/VizDoom-DRL-task2")
    
    if agent_type == "dqn":
        config = DQNConfig()
        if is_colab:
            # In Colab, we need to use the DQN class instead of DuelingDQN
            # since checkpoints were created with that architecture
            from agents.cnn_dqn.model import DQN
            model = DQN(state_dim=None, n_actions=env.action_space.n, 
                        frame_stack=config.frame_stack).to(config.device)
        else:
            # Local environment uses DuelingDQN
            model = DuelingDQN(env.action_space.n, config.frame_stack).to(config.device)
    elif agent_type == "ppo":
        config = PPOConfig()
        if is_colab:
            # Create a simplified PPO model for Colab that matches the architecture in the checkpoint
            class SimplePPOModel(nn.Module):
                def __init__(self, n_actions, frame_stack=4):
                    super(SimplePPOModel, self).__init__()
                    # Similar structure to the DQN model
                    self.feature_extractor = nn.Sequential(
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
                    
                    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
                    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
                    linear_input_size = convw * convh * 64
                    
                    # Actor (policy) head
                    self.actor = nn.Sequential(
                        nn.Linear(linear_input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, n_actions)
                    )
                    
                    # Critic (value) head
                    self.critic = nn.Sequential(
                        nn.Linear(linear_input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1)
                    )
                
                def forward(self, x):
                    # Handle input shape
                    if len(x.shape) == 5:  # [batch_size, 1, frame_stack, H, W]
                        x = x.squeeze(1)  # Remove the extra dimension
                    
                    features = self.feature_extractor(x)
                    
                    action_logits = self.actor(features)
                    action_probs = F.softmax(action_logits, dim=-1)
                    value = self.critic(features)
                    
                    return action_probs, value
                
                def predict(self, observation, deterministic=True):
                    with torch.no_grad():
                        if isinstance(observation, np.ndarray):
                            observation = torch.FloatTensor(observation).unsqueeze(0).to(next(self.parameters()).device)
                        
                        action_probs, _ = self.forward(observation)
                        
                        if deterministic:
                            # Take the action with highest probability
                            action = torch.argmax(action_probs, dim=1).item()
                        else:
                            # Sample from the probability distribution
                            action = torch.multinomial(action_probs.squeeze(), 1).item()
                            
                        return action, None  # Return None as second value to match expected interface
            
            model = SimplePPOModel(env.action_space.n, config.frame_stack).to(config.device)
        else:
            # Local environment uses TransformerPPO
            model = ActorCritic(
                env.action_space.n,
                frame_stack=config.frame_stack,
                resolution=config.resolution,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers
            ).to(config.device)
    elif agent_type == "sac":
        config = SACConfig()
        # For simplicity, we'll also use a simplified model for SAC in Colab
        if is_colab:
            # Similar to PPO, create a simplified SAC model
            class SimpleSACModel(nn.Module):
                def __init__(self, n_actions, frame_stack=4):
                    super(SimpleSACModel, self).__init__()
                    # Feature extraction (same as DQN)
                    self.feature_extractor = nn.Sequential(
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
                    
                    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
                    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
                    linear_input_size = convw * convh * 64
                    
                    # Actor network: outputs action probabilities
                    self.actor = nn.Sequential(
                        nn.Linear(linear_input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, n_actions)
                    )
                    
                    # Critic network: outputs Q-values
                    self.critic = nn.Sequential(
                        nn.Linear(linear_input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, n_actions)
                    )
                
                def forward(self, x):
                    # Handle input shape
                    if len(x.shape) == 5:  # [batch_size, 1, frame_stack, H, W]
                        x = x.squeeze(1)  # Remove the extra dimension
                    
                    features = self.feature_extractor(x)
                    
                    # Get action probabilities for actor
                    action_logits = self.actor(features)
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    # Get Q-values for critic
                    q_values = self.critic(features)
                    
                    return action_probs, q_values
                
                def predict(self, observation, deterministic=True):
                    with torch.no_grad():
                        if isinstance(observation, np.ndarray):
                            observation = torch.FloatTensor(observation).unsqueeze(0).to(next(self.parameters()).device)
                        
                        action_probs, _ = self.forward(observation)
                        
                        if deterministic:
                            # Take the action with highest probability
                            action = torch.argmax(action_probs, dim=1).item()
                        else:
                            # Sample from the probability distribution
                            action = torch.multinomial(action_probs.squeeze(), 1).item()
                            
                        return action, None
            
            model = SimpleSACModel(env.action_space.n, config.frame_stack).to(config.device)
        else:
            model = HybridSACNetwork(
                env.action_space.n,
                frame_stack=config.frame_stack,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers
            ).to(config.device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Handle policy_state_dict for Colab models
        if is_colab and "policy_state_dict" in checkpoint:
            print(f"Loading {agent_type} policy state dict")
            try:
                model.load_state_dict(checkpoint["policy_state_dict"])
                print("Successfully loaded policy state dict")
            except Exception as e:
                print(f"Error loading policy_state_dict directly: {e}")
                # Use a random model for evaluation when state dicts don't match
                print(f"Warning: Using randomly initialized {agent_type} model due to architecture mismatch")
        # Handle other checkpoint formats
        elif agent_type == "ppo" and "actor_state_dict" in checkpoint:
            print("Loading PPO actor/critic state dicts")
            model.actor.load_state_dict(checkpoint["actor_state_dict"])
            if "critic_state_dict" in checkpoint:
                model.critic.load_state_dict(checkpoint["critic_state_dict"])
        elif agent_type == "sac" and "actor_state_dict" in checkpoint:
            print("Loading SAC actor/critic state dicts")
            model.actor.load_state_dict(checkpoint["actor_state_dict"])
            if "critic_state_dict" in checkpoint:
                model.critic.load_state_dict(checkpoint["critic_state_dict"])
            if "critic_target_state_dict" in checkpoint:
                model.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        elif "model_state_dict" in checkpoint:
            print("Loading model state dict")
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            print("Loading state dict")
            model.load_state_dict(checkpoint["state_dict"])
        else:
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using randomly initialized model for evaluation")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized model for evaluation")
    
    model.eval()
    
    # Add predict method for models that don't have it
    if not hasattr(model, 'predict'):
        # Create a wrapper class that adds the predict method to our model
        class AgentWrapper:
            def __init__(self, model, device, agent_type):
                self.model = model
                self.device = device
                self.agent_type = agent_type
                
            def predict(self, observation, deterministic=True):
                with torch.no_grad():
                    if isinstance(observation, np.ndarray):
                        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    
                    # Different prediction logic based on model type
                    if self.agent_type == "dqn":
                        # DQN outputs q-values directly
                        q_values = self.model(observation)
                        action = torch.argmax(q_values, dim=1).item()
                    else:
                        # For other models, assume they return action probs as first output
                        outputs = self.model(observation)
                        if isinstance(outputs, tuple):
                            action_probs = outputs[0]  # First output is action probs
                            action = torch.argmax(action_probs, dim=1).item()
                        else:
                            # If model returns a single tensor, treat as q-values/logits
                            action = torch.argmax(outputs, dim=1).item()
                    
                    return action, None  # Return None as second value to match expected interface
        
        # Return the wrapped model
        return AgentWrapper(model, config.device, agent_type), config
    
    return model, config

def evaluate_all_agents(env, n_episodes=10, n_seeds=3):
    """Evaluate all agents across multiple episodes and seeds"""
    # Check if we're in Colab by looking at the file path in the error
    if os.path.exists("/content/VizDoom-DRL-task2"):
        checkpoint_dir = "/content/VizDoom-DRL-task2/checkpoints"
    else:
        # Local environment
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    # Print path for debugging
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found at {checkpoint_dir}")
        # Try to locate where checkpoints might be
        if os.path.exists("/content"):
            print("Contents of /content directory:")
            os.system("find /content -name '*.pt' | head -n 10")
    else:
        # List available checkpoint files
        checkpoint_files = os.listdir(checkpoint_dir)
        print(f"Available checkpoint files: {checkpoint_files}")
    
    # Select the best available checkpoints based on what's in the directory
    if os.path.exists(checkpoint_dir):
        # Find available checkpoints - Note that DQN files don't have 'cnn_' prefix in Colab
        dqn_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('dqn_episode_')]  # Changed from 'cnn_dqn_episode_'
        ppo_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('transformer_ppo_episode_')]
        sac_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('hybrid_sac_episode_')]
        
        # Choose the best checkpoints available
        dqn_file = max(dqn_files, key=lambda x: int(x.split('_')[-1].split('.')[0])) if dqn_files else "dqn_episode_0.pt"  # Changed default
        ppo_file = max(ppo_files, key=lambda x: int(x.split('_')[-1].split('.')[0])) if ppo_files else "transformer_ppo_episode_0.pt"
        sac_file = max(sac_files, key=lambda x: int(x.split('_')[-1].split('.')[0])) if sac_files else "hybrid_sac_episode_0.pt"
    else:
        # Default to episode 0 if directory doesn't exist
        dqn_file = "dqn_episode_0.pt"  # Changed default
        ppo_file = "transformer_ppo_episode_0.pt"
        sac_file = "hybrid_sac_episode_0.pt"
    
    # Setup agent paths
    agents = {
        "DQN": ("dqn", os.path.join(checkpoint_dir, dqn_file)),
        "PPO": ("ppo", os.path.join(checkpoint_dir, ppo_file)),
        "SAC": ("sac", os.path.join(checkpoint_dir, sac_file))
    }
    
    # Print chosen checkpoints
    for name, (_, path) in agents.items():
        print(f"Using {name} checkpoint: {path}")
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist!")
    
    results = {name: [] for name in agents.keys()}
    metrics = ['reward', 'steps', 'success_rate']
    
    for seed in range(n_seeds):
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)
        
        for agent_name, (agent_type, checkpoint_path) in agents.items():
            print(f"\nEvaluating {agent_name} (Seed {seed + 1}/{n_seeds})")
            model, config = load_agent(agent_type, checkpoint_path, env)
            
            # Evaluate agent
            eval_results = evaluate_agent(env, model, n_episodes=n_episodes)
            results[agent_name].append(eval_results)
            
    return results

def plot_comparison(results, save_path="comparison_results.png"):
    """Plot comparison of agent performances"""
    # Set up the plot style
    sns.set_style("whitegrid")
    
    # Prepare data for plotting
    metrics = ['mean_reward', 'mean_steps', 'std_reward', 'std_steps']
    agents = list(results.keys())
    
    # Verify we have data for all agents
    if not all(agents):
        print("Warning: No data available for some agents, skipping plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Agent Performance Comparison", fontsize=16)
    
    # Function to safely calculate mean and std
    def safe_stats(data_list, key):
        try:
            values = [r[key] for r in data_list if key in r]
            if not values:
                return 0, 0  # Default if no values
            return np.mean(values), np.std(values)
        except Exception as e:
            print(f"Error calculating stats for {key}: {e}")
            return 0, 0
    
    # Calculate stats for each agent
    mean_rewards = []
    std_rewards = []
    mean_steps = []
    std_steps = []
    
    for agent in agents:
        if not results[agent]:  # Skip if no data
            mean_rewards.append(0)
            std_rewards.append(0)
            mean_steps.append(0)
            std_steps.append(0)
            continue
            
        # Calculate means of means and means of stds
        mean_reward, err_reward = safe_stats([r for r in results[agent] if 'mean_reward' in r], 'mean_reward')
        mean_rewards.append(mean_reward)
        
        # For std, we use the mean of std values rather than std of means
        mean_std_reward, _ = safe_stats([r for r in results[agent] if 'std_reward' in r], 'std_reward')
        std_rewards.append(mean_std_reward)
        
        mean_step, err_step = safe_stats([r for r in results[agent] if 'mean_steps' in r], 'mean_steps')
        mean_steps.append(mean_step)
        
        mean_std_step, _ = safe_stats([r for r in results[agent] if 'std_steps' in r], 'std_steps')
        std_steps.append(mean_std_step)
    
    # Plot mean reward (without error bars if there's an issue)
    try:
        sns.barplot(x=agents, y=mean_rewards, ax=axes[0, 0])
        axes[0, 0].set_title("Average Reward")
        axes[0, 0].set_ylabel("Reward")
    except Exception as e:
        print(f"Error plotting mean rewards: {e}")
        # Fallback to simple bar chart
        axes[0, 0].bar(agents, mean_rewards)
        axes[0, 0].set_title("Average Reward")
        axes[0, 0].set_ylabel("Reward")
    
    # Plot mean steps (without error bars if there's an issue)
    try:
        sns.barplot(x=agents, y=mean_steps, ax=axes[0, 1])
        axes[0, 1].set_title("Average Steps per Episode")
        axes[0, 1].set_ylabel("Steps")
    except Exception as e:
        print(f"Error plotting mean steps: {e}")
        # Fallback to simple bar chart
        axes[0, 1].bar(agents, mean_steps)
        axes[0, 1].set_title("Average Steps per Episode")
        axes[0, 1].set_ylabel("Steps")
    
    # Plot reward consistency (std)
    axes[1, 0].bar(agents, std_rewards)
    axes[1, 0].set_title("Reward Consistency (Lower is Better)")
    axes[1, 0].set_ylabel("Standard Deviation")
    
    # Plot steps consistency (std)
    axes[1, 1].bar(agents, std_steps)
    axes[1, 1].set_title("Steps Consistency (Lower is Better)")
    axes[1, 1].set_ylabel("Standard Deviation")
    
    # Adjust layout and save
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

def main():
    # Create environment
    config = DQNConfig()  # Use any config, they share the same env settings
    env = VizDoomWrapper(
        config.scenario_path,
        frame_skip=config.frame_skip,
        frame_stack=config.frame_stack,
        resolution=config.resolution
    )
    
    # Evaluate all agents
    results = evaluate_all_agents(env, n_episodes=10, n_seeds=3)
    
    # Plot and save results
    plot_comparison(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for agent_name, agent_results in results.items():
        mean_reward = np.mean([r['mean_reward'] for r in agent_results])
        std_reward = np.mean([r['std_reward'] for r in agent_results])
        mean_steps = np.mean([r['mean_steps'] for r in agent_results])
        std_steps = np.mean([r['std_steps'] for r in agent_results])
        
        print(f"\n{agent_name}:")
        print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Steps: {mean_steps:.2f} ± {std_steps:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()