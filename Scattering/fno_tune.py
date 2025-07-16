import argparse
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
import numpy as np
from tqdm import tqdm
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
import os

print("Current working directory:", os.getcwd())
print("Files:", os.listdir())

class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}

parser = argparse.ArgumentParser()
parser.add_argument("--trial_id", type=int, required=True)
args = parser.parse_args()
trial = args.trial_id
random.seed(trial)
np.random.seed(trial)
torch.manual_seed(trial)
torch.cuda.manual_seed_all(trial)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

param_grid = {
    "n_modes": [(8, 8), (12, 12), (16, 16)],
    "hidden_channels": [16, 32, 64],
    "n_layers": [2, 3, 4],
    "lr": [1e-3, 3e-3, 8e-3, 1e-2],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "scheduler_type": ["ReduceLROnPlateau", "StepLR"]
}

num_epochs = 2000
patience = 30
val_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models_cnnb", exist_ok=True)

data = torch.load("farfield_improved.pt")
full_dataset = DictDataset(data["farfield"], data["farfield_improved"])

trial_params = {
    "n_modes": random.choice(param_grid["n_modes"]),
    "hidden_channels": random.choice(param_grid["hidden_channels"]),
    "n_layers": random.choice(param_grid["n_layers"]),
    "lr": random.choice(param_grid["lr"]),
    "weight_decay": random.choice(param_grid["weight_decay"]),
    "batch_size": random.choice(param_grid["batch_size"]),
    "scheduler_type": random.choice(param_grid["scheduler_type"]),
}

val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=trial_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=trial_params["batch_size"])

wandb.init(
    project="cnnb_tune",
    name=f"trial_{trial}",
    config={**trial_params, "val_split": val_split, "epochs": num_epochs, "patience": patience},
    reinit=True
)

model = FNO(
    n_modes=trial_params["n_modes"],
    in_channels=2,
    out_channels=2,
    hidden_channels=trial_params["hidden_channels"],
    n_layers=trial_params["n_layers"],
    projection_channel_ratio=2
).to(device)

wandb.watch(model, log="all", log_freq=10)

optimizer = AdamW(model.parameters(), lr=trial_params["lr"], weight_decay=trial_params["weight_decay"])

if trial_params["scheduler_type"] == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=20)
elif trial_params["scheduler_type"] == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()

    total_train_loss = 0.0
    total_train_h1_rel = 0.0
    total_train_h1_abs = 0.0
    total_train_l2_rel = 0.0
    total_train_l2_abs = 0.0

    for batch in tqdm(train_loader, desc=f"[Trial {trial}] Epoch {epoch+1}", leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        pred = model(x)

        h1_rel = h1loss.rel(pred, y)
        h1_abs = h1loss.abs(pred, y)
        l2_rel = l2loss.rel(pred, y)
        l2_abs = l2loss.abs(pred, y)
        
        loss = l2_rel
        if torch.isnan(loss):
            print("NaN loss detected. Skipping epoch")
            break
        loss.backward()
        optimizer.step()
        
        total_train_h1_rel += h1_rel.item() * x.size(0)
        total_train_h1_abs += h1_abs.item() * x.size(0)
        total_train_l2_rel += l2_rel.item() * x.size(0)
        total_train_l2_abs += l2_abs.item() * x.size(0)

        total_train_loss += loss.item() * x.size(0)

    train_size = len(train_loader.dataset)
    h1_rel_loss = total_train_h1_rel / train_size
    h1_abs_loss = total_train_h1_abs / train_size
    l2_rel_loss = total_train_l2_rel / train_size
    l2_abs_loss = total_train_l2_abs / train_size

    train_loss = h1_rel_loss

    model.eval()
    total_val_loss = 0.0
    total_val_h1_rel = 0.0
    total_val_h1_abs = 0.0
    total_val_l2_rel = 0.0
    total_val_l2_abs = 0.0
    
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            pred = model(x)
            
            h1_rel = h1loss.rel(pred, y)
            h1_abs = h1loss.abs(pred, y)
            l2_rel = l2loss.rel(pred, y)
            l2_abs = l2loss.abs(pred, y)

            val_loss = l2_rel

            total_val_h1_rel += h1_rel.item() * x.size(0)
            total_val_h1_abs += h1_abs.item() * x.size(0)
            total_val_l2_rel += l2_rel.item() * x.size(0)
            total_val_l2_abs += l2_abs.item() * x.size(0)

            total_val_loss += val_loss.item() * x.size(0)
            total_samples += x.size(0)

    h1_rel_loss_val = total_val_h1_rel / total_samples
    h1_abs_loss_val = total_val_h1_abs / total_samples
    l2_rel_loss_val = total_val_l2_rel / total_samples
    l2_abs_loss_val = total_val_l2_abs / total_samples

    val_loss_avg = l2_rel_loss_val

    wandb.log({
        "epoch": epoch + 1,
        "train/L2_abs": l2_abs_loss,
        "train/L2_rel": l2_rel_loss,
        "train/H1_abs": h1_abs_loss,
        "train/H1_rel": h1_rel_loss,
        "val/L2_abs": l2_abs_loss_val,
        "val/L2_rel": l2_rel_loss_val,
        "val/H1_abs": h1_abs_loss_val,
        "val/H1_rel": h1_rel_loss_val,
        "lr": scheduler.optimizer.param_groups[0]["lr"]
    })

    if trial_params["scheduler_type"] == "ReduceLROnPlateau":
        scheduler.step(val_loss_avg)
    else:
        scheduler.step()

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        model_path = f"models_cnnb/cnnb_trial_{trial}.pt"
        torch.save(model.state_dict(), model_path)
        artifact = wandb.Artifact(f"cnnb_model_trial_{trial}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"[Trial {trial}] Early stopping at epoch {epoch+1}")
            break

wandb.finish()
