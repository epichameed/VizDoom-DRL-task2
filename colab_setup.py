import os
import shutil
from google.colab import drive

def setup_colab():
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Create necessary directories
    os.makedirs('/content/VizDoom-DRL-task2', exist_ok=True)
    os.makedirs('/content/VizDoom-DRL-task2/scenarios', exist_ok=True)
    os.makedirs('/content/VizDoom-DRL-task2/checkpoints', exist_ok=True)
    
    # Install required packages
    !pip install vizdoom
    !pip install gymnasium
    !pip install wandb
    !pip install torch torchvision torchaudio
    !pip install opencv-python
    !pip install tqdm
    
    # Download scenario files
    !wget https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/basic.wad -O /content/VizDoom-DRL-task2/scenarios/basic.wad
    !wget https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/basic.cfg -O /content/VizDoom-DRL-task2/scenarios/basic.cfg
    
    print("Setup complete! You can now run your training script.")

if __name__ == "__main__":
    setup_colab() 