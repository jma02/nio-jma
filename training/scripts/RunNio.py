#!/usr/bin/env python3
"""
Main training script for NIO models.
"""
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration and utilities
try:
    from training.scripts.config import Config
    from training.scripts.model_factory import ModelFactory
    from training.scripts.trainer import Trainer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you have installed the package in development mode with 'pip install -e .'")
    sys.exit(1)

def setup_environment():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    setup_environment()
    
    if len(sys.argv) < 4:
        print("Usage: python RunNio.py <folder> <problem> <mod> <max_workers>")
        print("Example: python RunNio.py example helm nio_new 0")
        sys.exit(1)
    
    folder = sys.argv[1]
    problem = sys.argv[2]
    mod = sys.argv[3]
    max_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    # Create default config with provided arguments
    config = Config.from_args(folder, problem, mod, max_workers)
    config.save_config()
    
    # Initialize problem-specific dataset
    ProblemClass = config.get_problem_class()
    train_dataset = ProblemClass(
        norm=config.training_properties["norm"],
        inputs_bool=config.training_properties["inputs"],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        which="training",
        mod=config.mod
    )
    test_dataset = ProblemClass(
        norm=config.training_properties["norm"],
        inputs_bool=config.training_properties["inputs"],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        which="validation",
        mod=config.mod,
        noise=0.1
    )
    
    # Initialize model
    model = ModelFactory.create_model(
        config.problem,
        config.mod,
        train_dataset.inp_dim_branch,
        train_dataset.get_grid().squeeze(0).shape,
        config.branch_architecture,
        config.trunk_architecture,
        config.fno_architecture
    )
    
    # Load existing model if it exists
    if os.path.isfile(os.path.join(folder, "model.pkl")):
        print("Loading existing model")
        model.load_state_dict(torch.load(os.path.join(folder, "model.pkl")))
    
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Initialize and run trainer
    trainer = Trainer(config, model, train_dataset, test_dataset)
    trainer.train()

if __name__ == "__main__":
    main()
