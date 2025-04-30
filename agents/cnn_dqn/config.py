class Config:
    def __init__(self):
        # Environment settings
        self.scenario_path = "scenarios/basic.cfg"
        self.frame_stack = 4
        self.frame_skip = 4
        self.resolution = (84, 84)
        
        # Training settings
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
        
        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model architecture
        self.use_dueling = True
        self.hidden_size = 512
        
        # Logging settings
        self.log_interval = 10
        self.eval_interval = 100
        self.eval_episodes = 10
        
        # Random seeds
        self.seeds = [42, 123, 456]
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')} 