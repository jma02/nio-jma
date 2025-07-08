import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.debug_tools import CudaMemoryDebugger

class Trainer:
    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cuda_debugger = CudaMemoryDebugger(False)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training_properties["batch_size"],
            shuffle=True,
            num_workers=config.max_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.training_properties["batch_size"],
            shuffle=False,
            num_workers=config.max_workers,
            pin_memory=True
        )
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.training_properties["learning_rate"],
            weight_decay=config.training_properties["weight_decay"]
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.training_properties["step_size"],
            gamma=config.training_properties["gamma"]
        )
        
        self.writer = SummaryWriter(config.folder)
        
    def train(self):
        counter = 0
        best_loss = float('inf')
        
        # Get grid from the dataset
        grid = self.train_dataset.get_grid().to(self.device)
        
        for epoch in range(self.config.training_properties["epochs"]):
            self.model.train()
            train_mse = 0.0
            running_relative_train_mse = 0.0
            
            with tqdm(unit="batch", disable=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                for batch_idx, (inputs, target) in enumerate(self.train_loader):
                    inputs, target = inputs.to(self.device), target.to(self.device)
                    
                    # Expand grid to match batch size
                    batch_size = inputs.size(0)
                    batch_grid = grid.expand(batch_size, *grid.shape[1:]).to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(inputs, batch_grid)
                    loss = torch.nn.functional.mse_loss(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_mse += loss.item()
                    relative_mse = torch.mean(torch.square((output - target) / (target + 1e-8))).item()
                    running_relative_train_mse += relative_mse
                    
                    if batch_idx % 10 == 0:
                        tepoch.set_postfix(loss=loss.item())
                
                train_mse /= len(self.train_loader)
                running_relative_train_mse /= len(self.train_loader)
                
                self.scheduler.step()
                
                # Validation
                self.model.eval()
                test_mse = 0.0
                running_relative_test_mse = 0.0
                
                with torch.no_grad():
                    for batch_idx, (inputs, target) in enumerate(self.test_loader):
                        inputs, target = inputs.to(self.device), target.to(self.device)
                        batch_size = inputs.size(0)
                        batch_grid = grid.expand(batch_size, *grid.shape[1:]).to(self.device)
                        output = self.model(inputs, batch_grid)
                        loss = torch.nn.functional.mse_loss(output, target)
                        test_mse += loss.item()
                        relative_mse = torch.mean(torch.square((output - target) / (target + 1e-8))).item()
                        running_relative_test_mse += relative_mse
                
                test_mse /= len(self.test_loader)
                running_relative_test_mse /= len(self.test_loader)
                
                self.writer.add_scalar('Loss/train', train_mse, epoch)
                self.writer.add_scalar('Loss/test', test_mse, epoch)
                self.writer.add_scalar('Relative MSE/train', running_relative_train_mse, epoch)
                self.writer.add_scalar('Relative MSE/test', running_relative_test_mse, epoch)
                
                if test_mse < best_loss:
                    best_loss = test_mse
                    counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.config.folder, "model.pkl"))
                else:
                    counter += 1
                
                if counter > patience:
                    print(f"Early stopping at epoch {epoch} after {patience} epochs without improvement")
                    break
        
        self.writer.flush()
        self.writer.close()
