# ViZDoom DRL Task 2

This project implements a Deep Q-Network (DQN) agent to play ViZDoom using deep reinforcement learning.

## Project Structure

```
VizDoom-DRL-task2/
├── agents/
│   ├── cnn_dqn/
│   │   ├── model.py      # DQN model architecture
│   │   └── train.py      # Training script
│   └── common/
│       └── utils.py      # Environment wrapper and utilities
├── scenarios/            # ViZDoom scenario files
└── checkpoints/         # Saved model checkpoints
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/VizDoom-DRL-task2.git
cd VizDoom-DRL-task2
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download scenario files:

```bash
python download_scenarios.py
```

## Training

To train the DQN agent:

```bash
python -m agents.cnn_dqn.train
```

## Google Colab

You can also train the agent using Google Colab's GPU resources. The notebook `vizdoom_drl_colab.ipynb` contains all the necessary setup and training code.

## Requirements

- Python 3.8+
- PyTorch
- ViZDoom
- Gymnasium
- Wandb (for logging)
- OpenCV
- NumPy
