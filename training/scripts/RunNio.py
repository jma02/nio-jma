import sys
import os
import random
import numpy as np
import torch
from .config import Config
from .model_factory import ModelFactory
from .trainer import Trainer

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
    
    folder = sys.argv[1]
    config = Config.from_args(folder, *sys.argv[2:])
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
