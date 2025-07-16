import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import pandas as pd
import h5py
from scipy.sparse.linalg import lsqr
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import cmocean as cmo
from neuralop import LpLoss

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

folder = Path("U Crime")
folder.mkdir(parents=True, exist_ok=True)

farf_file_path = "/home/adesai/data/scattering/data_gen_final/farftest.hdf5"


with h5py.File(farf_file_path, "r") as hdf_file:
    image = hdf_file["image"][:]
    farfield_real = hdf_file["farfield.real"][:]
    farfield_imag = hdf_file["farfield.imag"][:]

N_TEST_SAMPLES = farfield_real.shape[1]
image_Ngrid = 100
Ngrid = 100
nfar = int(np.sqrt(farfield_real.shape[0]))
images = image.T.reshape(N_TEST_SAMPLES, image_Ngrid, image_Ngrid)
farfield_real = farfield_real.T.reshape(N_TEST_SAMPLES, nfar, nfar)
farfield_imag = farfield_imag.T.reshape(N_TEST_SAMPLES, nfar, nfar)

farfields = farfield_real + 1J*farfield_imag

farfield = np.stack([farfield_real, farfield_imag], axis=1) # Shape: (N_TEST_SAMPLES, 2, nfar, nfar)
print(farfield.shape)
farfield_tensor = torch.tensor(farfield, dtype=torch.float32)

# Set device to GPU if available
device = torch.device('cuda')
print("Using device:", device)

farfield_tensor = farfield_tensor.to(device)

test_dataset1 = TensorDataset(farfield_tensor)

test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNModel, self).__init__()

        # Initialize CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = 2

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Compute the flatten size dynamically after CNN layers
        self.flatten_size = self._compute_flatten_size(input_shape)

        # Initialize FC layers
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        # Final output layer with linear activation
        self.output_layer = nn.Linear(input_fc_dim, 100 * 100)

        # Activation function saved
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        # Pass through CNN layers
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)  # Conv
            x = self.activation_fn(x)  # Apply activation function only here
            x = self.cnn_layers[i + 1](x)  # MP
            x = self.cnn_layers[i + 2](x)  # DP

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)  # Fc layer
            x = self.activation_fn(x)  # Activation
            x = self.fc_layers[i + 1](x)  # Dps

        # Final linear output layer
        x = self.output_layer(x)

        # Reshape to output shape
        x = x.view(x.size(0), 100, 100)
        return x

class BCNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(BCNNModel, self).__init__()

        # Initialize CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Compute the flatten size dynamically after CNN layers
        self.flatten_size = self._compute_flatten_size(input_shape)

        # Initialize FC layers
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        # Final output layer with linear activation
        self.output_layer = nn.Linear(input_fc_dim, output_shape[0] * output_shape[1])

        # Activation function saved
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        # Pass through CNN layers
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)  # Conv
            x = self.activation_fn(x)  # Apply activation function only here
            x = self.cnn_layers[i + 1](x)  # MP
            x = self.cnn_layers[i + 2](x)  # DP

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)  # Fc layer
            x = self.activation_fn(x)  # Activation
            x = self.fc_layers[i + 1](x)  # Dps

        # Final linear output layer
        x = self.output_layer(x)

        # Reshape to output shape
        x = x.view(x.size(0), 100, 100)
        return x

class CNNBModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNBModel, self).__init__()

        self.output_shape = output_shape  # Store output shape

        # Initialize CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Compute the flatten size dynamically after CNN layers
        self.flatten_size = self._compute_flatten_size(input_shape)

        # Initialize FC layers
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        # Final output layer with flattened output shape
        self.output_layer = nn.Linear(input_fc_dim, int(np.prod(output_shape)))

        # Save activation function
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        # Pass through CNN layers
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)  # Conv
            x = self.activation_fn(x)  # Activation
            x = self.cnn_layers[i + 1](x)  # MaxPool
            x = self.cnn_layers[i + 2](x)  # Dropout

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)  # Fully connected layer
            x = self.activation_fn(x)  # Activation
            x = self.fc_layers[i + 1](x)  # Dropout

        # Final linear output layer
        x = self.output_layer(x)

        # Reshape to output shape
        x = x.view(-1, *self.output_shape)

        return x

input_shape1 = (2, nfar, nfar)  # PyTorch uses channels first
input_shape2 = (1, Ngrid, Ngrid)

output_shape1 = (2, nfar, nfar)
output_shape2 = (Ngrid, Ngrid)

num_cnn_layers = 4
channels_per_layer = [125, 358, 426, 221]
num_fc_layers = 1
fc_units = [576]
activation_function = nn.GELU()
dropout_rate = 0

CNNB_opt = CNNBModel(input_shape1, output_shape1, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_function, dropout_rate)

num_cnn_layers = 4
channels_per_layer = [335, 33, 195, 65]
num_fc_layers = 1
fc_units = [971]
activation_function = nn.GELU()
dropout_rate = 0

BCNN_opt = BCNNModel(input_shape1, output_shape2, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_function, dropout_rate)

num_cnn_layers = 4
channels_per_layer = [296, 211, 152, 61]
num_fc_layers = 3
fc_units = [537, 465, 419]
activation_function = nn.GELU()
dropout_rate = 0

CNN_opt = CNNModel(input_shape1, output_shape2, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_function, dropout_rate)

CNNB_opt = CNNB_opt.to(device)
BCNN_opt = BCNN_opt.to(device)
CNN_opt = CNN_opt.to(device)

CNNB_opt.load_state_dict(torch.load('CNNB_tuned_model_kvalue1.pt'))
BCNN_opt.load_state_dict(torch.load('BCNN_tuned_model_kvalue1.pt'))
CNN_opt.load_state_dict(torch.load('CNN_tuned_model_kvalue1.pt'))

CNNB_opt.eval()
BCNN_opt.eval()
CNN_opt.eval()

CNNB_opt_preds = []
BCNN_opt_preds = []
CNN_opt_preds = []

print("CNNB Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader1, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = CNNB_opt(inputs)
        CNNB_opt_preds.append(outputs.cpu().numpy())
CNNB_opt_preds = np.concatenate(CNNB_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, nfar, nfar)

print("BCNN Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader1, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = BCNN_opt(inputs)
        BCNN_opt_preds.append(outputs.cpu().numpy())
BCNN_opt_preds = np.concatenate(BCNN_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, nfar, nfar)

print("CNN Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader1, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = CNN_opt(inputs)
        CNN_opt_preds.append(outputs.cpu().numpy())
CNN_opt_preds = np.concatenate(CNN_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, Ngrid, Ngrid)

# Build Born Matrix
def discretize_born(k, xlim, phi, Ngrid, theta):
    vert_step = 2 * xlim / Ngrid
    hor_step = 2 * xlim / Ngrid
    Cfac = vert_step * hor_step * np.exp(1j * np.pi / 4) * np.sqrt(k**3 / (np.pi * 8))
    y1 = np.linspace(-xlim, xlim, Ngrid)
    y2 = np.linspace(-xlim, xlim, Ngrid)
    Y1, Y2 = np.meshgrid(y1, y2)
    grid_points = np.column_stack((Y1.ravel(), Y2.ravel()))
    xhat = np.array([np.cos(theta), np.sin(theta)]).T
    d = np.array([np.cos(phi), np.sin(phi)])
    diff = xhat - d
    dot_products = np.dot(diff, grid_points.T)
    Exp = np.exp(1j * k * dot_products)
    A = Cfac * Exp
    return A

def build_born(incp, farp, kappa, xlim, Ngrid):
    phi=np.zeros(incp["n"])
    center=incp["cent"]
    app=incp["app"]
    for ip in range(0,incp["n"]):
        if incp["n"]==1:
            phi[0]=0.0
        else:
            phi[ip]=(center-app/2)+app*ip/(incp["n"]-1)
    ntheta=farp["n"]
    ctheta=farp["cent"]
    apptheta=farp["app"]
    theta=np.zeros(ntheta)
    for jp in range(0,ntheta):
        theta[jp]=(ctheta-apptheta/2)+apptheta*jp/(ntheta-1)
    born_operator_list = [discretize_born(kappa, xlim, inc_field, Ngrid, theta) for inc_field in phi]
    operator_combined = np.vstack(born_operator_list)
    return operator_combined

ninc = 100
nfar = 100
incp = {"n": ninc, "app": 2*np.pi, "cent":0}
farp = {"n": nfar, "app": 2*np.pi, "cent":0}
kappa = 16
Ngrid = 100
xlim = 1
born = build_born(incp, farp, kappa, xlim, Ngrid)

ticks = [0, 50, 99]
tick_labels = [r"$-1$", r"$0$", r"$1$"]

l2_rel_loss = LpLoss(d=2, p=2, reduction='mean', measure=4.0)
l1_rel_loss = LpLoss(d=2, p=1, reduction='mean', measure=4.0)

l2_abs_loss = LpLoss(d=2, p=2, reduction='mean', measure=4.0)
l1_abs_loss = LpLoss(d=2, p=1, reduction='mean', measure=4.0)

l2error_rel = {"Born1": [], "Born2": [], "CNNB": [], "BCNN": [], "CNN": []}
l1error_rel = {"Born1": [], "Born2": [], "CNNB": [], "BCNN": [], "CNN": []}
l2error_abs = {"Born1": [], "Born2": [], "CNNB": [], "BCNN": [], "CNN": []}
l1error_abs = {"Born1": [], "Born2": [], "CNNB": [], "BCNN": [], "CNN": []}

for i in range(N_TEST_SAMPLES):
    print("Sample: ", i)
    ground_truth = images[i].flatten()
    true_farfield =  farfields[i].reshape(nfar*nfar,1)
    CNNB_opt_pred = (CNNB_opt_preds[i][0]+1J*CNNB_opt_preds[i][1]).reshape(nfar*nfar,1)
    BCNN_opt_pred = BCNN_opt_preds[i].reshape(Ngrid,Ngrid)
    CNN_opt_pred = CNN_opt_preds[i].reshape(Ngrid,Ngrid)
    born1 = np.real(lsqr(born, true_farfield, damp=1e0)[0]).flatten()
    born2 = np.real(lsqr(born, true_farfield, damp=1e-1)[0]).flatten()
    nn1 = np.real(lsqr(born, CNNB_opt_pred, damp=1e-1)[0]).flatten()
    nn2 = BCNN_opt_pred.flatten()+born1
    nn3 = CNN_opt_pred.flatten()

    gt_tensor = torch.tensor(ground_truth.reshape(100, 100), dtype=torch.float32, device=device)
    born1_tensor = torch.tensor(born1.reshape(100, 100), dtype=torch.float32, device=device)
    born2_tensor = torch.tensor(born2.reshape(100, 100), dtype=torch.float32, device=device)
    nn1_tensor = torch.tensor(nn1.reshape(100, 100), dtype=torch.float32, device=device)
    nn2_tensor = torch.tensor(nn2.reshape(100, 100), dtype=torch.float32, device=device)
    nn3_tensor = torch.tensor(nn3.reshape(100, 100), dtype=torch.float32, device=device)

    # L2 relative
    l2error_rel["Born1"].append(l2_rel_loss.rel(born1_tensor, gt_tensor).item())
    l2error_rel["Born2"].append(l2_rel_loss.rel(born2_tensor, gt_tensor).item())
    l2error_rel["CNNB"].append(l2_rel_loss.rel(nn1_tensor, gt_tensor).item())
    l2error_rel["BCNN"].append(l2_rel_loss.rel(nn2_tensor, gt_tensor).item())
    l2error_rel["CNN"].append(l2_rel_loss.rel(nn3_tensor, gt_tensor).item())

    # L1 relative
    l1error_rel["Born1"].append(l1_rel_loss.rel(born1_tensor, gt_tensor).item())
    l1error_rel["Born2"].append(l1_rel_loss.rel(born2_tensor, gt_tensor).item())
    l1error_rel["CNNB"].append(l1_rel_loss.rel(nn1_tensor, gt_tensor).item())
    l1error_rel["BCNN"].append(l1_rel_loss.rel(nn2_tensor, gt_tensor).item())
    l1error_rel["CNN"].append(l1_rel_loss.rel(nn3_tensor, gt_tensor).item())

    # L2 absolute
    l2error_abs["Born1"].append(l2_abs_loss.abs(born1_tensor, gt_tensor).item())
    l2error_abs["Born2"].append(l2_abs_loss.abs(born2_tensor, gt_tensor).item())
    l2error_abs["CNNB"].append(l2_abs_loss.abs(nn1_tensor, gt_tensor).item())
    l2error_abs["BCNN"].append(l2_abs_loss.abs(nn2_tensor, gt_tensor).item())
    l2error_abs["CNN"].append(l2_abs_loss.abs(nn3_tensor, gt_tensor).item())

    # L1 absolute
    l1error_abs["Born1"].append(l1_abs_loss.abs(born1_tensor, gt_tensor).item())
    l1error_abs["Born2"].append(l1_abs_loss.abs(born2_tensor, gt_tensor).item())
    l1error_abs["CNNB"].append(l1_abs_loss.abs(nn1_tensor, gt_tensor).item())
    l1error_abs["BCNN"].append(l1_abs_loss.abs(nn2_tensor, gt_tensor).item())
    l1error_abs["CNN"].append(l1_abs_loss.abs(nn3_tensor, gt_tensor).item())

    print(f"\nSample {i}:")
    print(f"  Relative L2:")
    print(f"    Born1 : {l2error_rel['Born1'][-1]:.5e}")
    print(f"    Born2 : {l2error_rel['Born2'][-1]:.5e}")
    print(f"    CNNB  : {l2error_rel['CNNB'][-1]:.5e}")
    print(f"    BCNN  : {l2error_rel['BCNN'][-1]:.5e}")
    print(f"    CNN   : {l2error_rel['CNN'][-1]:.5e}")

    print(f"  Relative L1:")
    print(f"    Born1 : {l1error_rel['Born1'][-1]:.5e}")
    print(f"    Born2 : {l1error_rel['Born2'][-1]:.5e}")
    print(f"    CNNB  : {l1error_rel['CNNB'][-1]:.5e}")
    print(f"    BCNN  : {l1error_rel['BCNN'][-1]:.5e}")
    print(f"    CNN   : {l1error_rel['CNN'][-1]:.5e}")

    print(f"  Absolute L2:")
    print(f"    Born1 : {l2error_abs['Born1'][-1]:.5e}")
    print(f"    Born2 : {l2error_abs['Born2'][-1]:.5e}")
    print(f"    CNNB  : {l2error_abs['CNNB'][-1]:.5e}")
    print(f"    BCNN  : {l2error_abs['BCNN'][-1]:.5e}")
    print(f"    CNN   : {l2error_abs['CNN'][-1]:.5e}")

    print(f"  Absolute L1:")
    print(f"    Born1 : {l1error_abs['Born1'][-1]:.5e}")
    print(f"    Born2 : {l1error_abs['Born2'][-1]:.5e}")
    print(f"    CNNB  : {l1error_abs['CNNB'][-1]:.5e}")
    print(f"    BCNN  : {l1error_abs['BCNN'][-1]:.5e}")
    print(f"    CNN   : {l1error_abs['CNN'][-1]:.5e}")


    # nn1 = nn1.reshape(100, 100)
    # nn2 = nn2.reshape(100, 100)
    # nn3 = nn3.reshape(100, 100)
    # born1 = born1.reshape(100, 100)
    # born2 = born2.reshape(100, 100)
    # ground_truth = ground_truth.reshape(100, 100)

    #baseline = [ground_truth+1, born1+1, born2+1]
    # baseline_titles = ["(I.) Ground Truth", r"(II.) Born ($\gamma=1$)", r"(III.) Born ($\gamma=1/10$)"]
    # nndata = [nn1+1,nn2+1,nn3+1]
    # nndata_titles = ["(IV.) CNNB", "(V.) BCNN", "(VI.) CNN"]

    # vmin = .8
    # vmax = 2
    # print(np.real(ground_truth).max())

    # # Plot results
    # fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=False)

    # # Reconstructions and true scatterer
    # for j in range(3):
    #     im1 = axes[0, j].imshow(baseline[j], origin="lower", cmap='cmo.dense', vmin=vmin, vmax=vmax)
    #     axes[0, j].set_xlabel(baseline_titles[j], fontsize=12)
    #     if baseline_titles[j] == "(I.) Ground Truth":
    #         ticks = [0, 50, 99]
    #     else:
    #         ticks = [0, 50, 99]
    #     axes[0, j].set_xticks(ticks)
    #     axes[0, j].set_xticklabels(tick_labels)
    #     axes[0, j].set_yticks(ticks)
    #     axes[0, j].set_yticklabels(tick_labels)

    #     ticks = [0, 50, 99]
    #     im2 = axes[1, j].imshow(nndata[j], origin="lower", cmap='cmo.dense', vmin=vmin, vmax=vmax)
    #     axes[1, j].set_xlabel(nndata_titles[j], fontsize=12)
    #     axes[1, j].set_xticks(ticks)
    #     axes[1, j].set_xticklabels(tick_labels)
    #     axes[1, j].set_yticks(ticks)
    #     axes[1, j].set_yticklabels(tick_labels)
    # cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    # plt.savefig(folder / f'test_result_{i}.png', bbox_inches='tight')
    # plt.close(fig)

pd.DataFrame(l2error_rel).to_csv("l2_rel_error.csv", index=False)
pd.DataFrame(l1error_rel).to_csv("l1_rel_error.csv", index=False)
pd.DataFrame(l2error_abs).to_csv("l2_abs_error.csv", index=False)
pd.DataFrame(l1error_abs).to_csv("l1_abs_error.csv", index=False)
