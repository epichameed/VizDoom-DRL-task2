import os
import numpy as np
import torch
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
    if agent_type == "dqn":
        config = DQNConfig()
        model = DuelingDQN(env.action_space.n, config.frame_stack).to(config.device)
    elif agent_type == "ppo":
        config = PPOConfig()
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
        model = HybridSACNetwork(
            env.action_space.n,
            frame_stack=config.frame_stack,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers
        ).to(config.device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def evaluate_all_agents(env, n_episodes=10, n_seeds=3):
    """Evaluate all agents across multiple episodes and seeds"""
    agents = {
        "DQN": ("dqn", "checkpoints/cnn_dqn_episode_900.pt"),
        "PPO": ("ppo", "checkpoints/transformer_ppo_episode_900.pt"),
        "SAC": ("sac", "checkpoints/hybrid_sac_episode_900.pt")
    }
    
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
    plt.figure(figsize=(15, 10))
    
    # Prepare data for plotting
    metrics = ['mean_reward', 'mean_steps', 'std_reward', 'std_steps']
    agents = list(results.keys())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Agent Performance Comparison", fontsize=16)
    
    # Plot mean reward
    sns.barplot(
        x=agents,
        y=[np.mean([r['mean_reward'] for r in results[a]]) for a in agents],
        yerr=[np.std([r['mean_reward'] for r in results[a]]) for a in agents],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Average Reward")
    axes[0, 0].set_ylabel("Reward")
    
    # Plot mean steps
    sns.barplot(
        x=agents,
        y=[np.mean([r['mean_steps'] for r in results[a]]) for a in agents],
        yerr=[np.std([r['mean_steps'] for r in results[a]]) for a in agents],
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("Average Steps per Episode")
    axes[0, 1].set_ylabel("Steps")
    
    # Plot reward consistency (std)
    sns.barplot(
        x=agents,
        y=[np.mean([r['std_reward'] for r in results[a]]) for a in agents],
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Reward Consistency (Lower is Better)")
    axes[1, 0].set_ylabel("Standard Deviation")
    
    # Plot steps consistency (std)
    sns.barplot(
        x=agents,
        y=[np.mean([r['std_steps'] for r in results[a]]) for a in agents],
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Steps Consistency (Lower is Better)")
    axes[1, 1].set_ylabel("Standard Deviation")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
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