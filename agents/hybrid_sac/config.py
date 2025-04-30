import torch

class Config:
    def __init__(self):
        # Environment settings
        self.scenario_path = "scenarios/basic.cfg"
        self.frame_stack = 4
        self.frame_skip = 4
        self.resolution = (84, 84)
        
        # Model architecture
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.hidden_dim = 256
        
        # SAC parameters
        self.num_episodes = 1000
        self.max_steps = 1000
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.replay_capacity = 100000
        self.update_frequency = 4
        self.target_update_interval = 1
        self.automatic_entropy_tuning = True
        
        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Logging settings
        self.log_interval = 10
        self.eval_interval = 100
        self.eval_episodes = 10
        self.save_frequency = 100
        
        # Random seeds
        self.seeds = [42, 123, 456]
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')} 