import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from collections import deque

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
        
        # Initialize frame buffer
        self.frames = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            self.frames.append(np.zeros(resolution, dtype=np.uint8))
            
        # Get action space
        self.actions = np.identity(self.game.get_available_buttons_size(), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Get observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, *resolution),
            dtype=np.uint8
        )
        
    def seed(self, seed=None):
        """Set the seed for the environment"""
        if seed is not None:
            self.game.set_seed(seed)
        return [seed]
        
    def reset(self, seed=None):
        self.game.new_episode()
        state = self.game.get_state()
        frame = self.preprocess(state.screen_buffer)
        
        # Reset frame buffer
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
            
        return np.stack(self.frames), {}
        
    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        
        if not done:
            state = self.game.get_state()
            frame = self.preprocess(state.screen_buffer)
            self.frames.append(frame)
        else:
            frame = np.zeros(self.resolution, dtype=np.uint8)
            self.frames.append(frame)
            
        return np.stack(self.frames), reward, done, done, {}
        
    def preprocess(self, frame):
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        # Resize
        frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
        
        return frame
        
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