import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

class VizDoomWrapper(gym.Env):
    def __init__(self, scenario_path, frame_skip=4, frame_stack=4, resolution=(84, 84)):
        super().__init__()
        
        # Initialize ViZDoom game
        self.game = DoomGame()
        self.game.load_config(scenario_path)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()
        
        # Environment parameters
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.resolution = resolution
        
        # Define action space
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, resolution[0], resolution[1]),
            dtype=np.uint8
        )
        
        # Initialize frame buffer
        self.frame_buffer = np.zeros((frame_stack, *resolution), dtype=np.uint8)
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Resize
        frame = cv2.resize(frame, self.resolution)
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        return frame
        
    def reset(self, seed=None):
        self.game.new_episode()
        frame = self.game.get_state().screen_buffer
        frame = self.preprocess_frame(frame)
        
        # Initialize frame buffer
        self.frame_buffer = np.stack([frame] * self.frame_stack)
        return self.frame_buffer, {}
        
    def step(self, action):
        reward = 0
        # Convert action to list format expected by ViZDoom
        action_list = [0] * self.game.get_available_buttons_size()
        action_list[action] = 1
        
        for _ in range(self.frame_skip):
            reward += self.game.make_action(action_list)
            if self.game.is_episode_finished():
                break
                
        if self.game.is_episode_finished():
            return self.frame_buffer, reward, True, True, {}
            
        # Get new frame
        frame = self.game.get_state().screen_buffer
        frame = self.preprocess_frame(frame)
        
        # Update frame buffer
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = frame
        
        return self.frame_buffer, reward, False, False, {}
        
    def close(self):
        self.game.close()
        
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.game.get_state().screen_buffer
        return None

def evaluate_agent(env, agent, n_episodes=10):
    """Evaluate an agent over multiple episodes"""
    total_rewards = []
    total_steps = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps)
    } 