"""
Legacy entrypoint for the Transformer Encoder model.

This file has been refactored and split into:
- config.py (Hyperparameters and configurations)
- model.py (EncoderModel architecture)
- dataset.py (NewDataset data loader)
- trainer.py (Trainer class for the model)
- main.py (Entry point for execution)

This file now acts as a wrapper that imports and exposes the modular elements,
allowing legacy code to import from encoder.py directly, and executing the training
pipeline if run directly.
"""

import os
import sys

# Add current directory to path to allow direct execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main as run_main
from model import EncoderModel
from dataset import NewDataset
from config import ModelConfig, TrainingConfig
from trainer import Trainer

if __name__ == "__main__":
    run_main()