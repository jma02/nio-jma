import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets.BornFarField import BornFarFieldDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import cmocean

model = torch.load("born_farfield_run1/model.pkl", map_location=device)

norm = "log-minmax"
inputs_bool = 2
max_workers = 2

eval_dataset = BornFarFieldDataset(norm=norm, inputs_bool=inputs_bool, device=device, which="testing", mod="nio")
grid = eval_dataset.get_grid().squeeze(0).to(device)
eval_dataset = DataLoader(eval_dataset, batch_size=128, shuffle=True, num_workers=max_workers, pin_memory=True)

errors = []
relative_errors = []
model.eval()
with torch.no_grad():
    for i, (inp, target) in enumerate(eval_dataset):
        inp = inp.to(device)
        target = target.to(device)
        
        pred = model.forward(inp, grid)
        error = torch.nn.functional.mse_loss(pred, target, reduction='mean').item()
        rel_error = (torch.norm(pred - target) / torch.norm(target)).item()
        errors.append(error)
        relative_errors.append(rel_error)
        print(f"Batch {i}: MSE={error:.6f}, Relative Error={rel_error:.6f}")

          # Plot a couple from each batch
        n_plot = min(3, inp.shape[0])  # plot up to 3 samples per batch
        fig, axes = plt.subplots(n_plot, 2, figsize=(8, 3 * n_plot))
        for j in range(n_plot):
            axes[j, 0].imshow(target[j].cpu().numpy(), aspect='auto', cmap=cmocean.cm.ice)
            axes[j, 0].set_title(f"True Sample {j}")
            axes[j, 0].axis('off')
            axes[j, 1].imshow(pred[j].cpu().numpy(), aspect='auto', cmap=cmocean.cm.ice)
            axes[j, 1].set_title(f"Pred Sample {j}")
            axes[j, 1].axis('off')
        plt.tight_layout()
        plt.savefig(f"src/visualization/images/eval_farfield/eval_batch_{i}.png")
        plt.close()
print("Errors:", errors)
print("Relative Errors:", relative_errors)