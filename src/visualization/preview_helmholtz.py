import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datasets.HelmNIO import HelmNIODataset as MyDataset  # adjust this import if your file path differs
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = MyDataset(
    norm="none",       # or "none", "norm", etc.
    inputs_bool=2,     # as used in training
    device=device,
    which="training",
    mod="nio",         # or "fcnn", "don" etc.
    noise=0.0,
    samples=10         # just a few for visualization
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Plotting
for i, (inp, out) in enumerate(loader):
    inp = inp.squeeze().cpu().numpy()
    out = out.squeeze().cpu().numpy()
    print(f"Input shape: {inp.shape}, Output shape: {out.shape}")

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    # Input visualization
    if inp.ndim == 3:
        for j in range(min(inp.shape[0], 4)):  # show up to 4 input channels
            axs[j].imshow(inp[j], aspect='auto', cmap='viridis')
            axs[j].set_title(f"Input channel {j}")
            axs[j].axis('off')
    else:
        axs[0].imshow(inp, aspect='auto', cmap='viridis')
        axs[0].set_title("Input")
        axs[0].axis('off')

    # Output visualization
    axs[4].imshow(out, aspect='auto', cmap='magma')
    axs[4].set_title("Output")
    axs[4].axis('off')

    out_path = Path(f"src/visualization/images/helmholtz{i}.png")
    plt.savefig(out_path)

    if i >= 4:
        break  # Show only first 5 samples