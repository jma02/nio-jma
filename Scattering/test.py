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
from tqdm import tqdm  # For progress bars
import cmocean as cmo

# My GPU doesn't have latex.
plt.rc('text', usetex=False)
plt.rc('font', family='serif')


folder = Path("Scattering/Plots")
folder.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')
data = torch.load("Scattering/farfield_image_test.pt", map_location=device)
farfield = data["farfield"].to(device)
images = data["image"].to(device)

farfields = farfield[:, 0].cpu().numpy() + 1J*farfield[:, 1].cpu().numpy()

N_TEST_SAMPLES = 4000
image_Ngrid = 100
Ngrid = 100
nfar = 100

delta_noise = 0.0

farfield_tensor = torch.tensor(farfield, dtype=torch.float32)
farfield_tensor = farfield_tensor.to(device)
farfield_tensor = (1 + torch.randn(farfield_tensor.shape).to(device)*delta_noise) * farfield_tensor  # Add multiplicative noise
input_tensor = farfield_tensor.to(device)
test_dataset1 = TensorDataset(input_tensor)
test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNModel, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = 2

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.flatten_size = self._compute_flatten_size(input_shape)
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        self.output_layer = nn.Linear(input_fc_dim, 100 * 100)
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)
            x = self.activation_fn(x)
            x = self.cnn_layers[i + 1](x)
            x = self.cnn_layers[i + 2](x)

        x = x.view(x.size(0), -1)

        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)
            x = self.activation_fn(x)
            x = self.fc_layers[i + 1](x)
        x = self.output_layer(x)
        x = x.view(x.size(0), 100, 100)
        return x

class BCNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(BCNNModel, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        self.flatten_size = self._compute_flatten_size(input_shape)
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        self.output_layer = nn.Linear(input_fc_dim, output_shape[0] * output_shape[1])
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)
            x = self.activation_fn(x)
            x = self.cnn_layers[i + 1](x)
            x = self.cnn_layers[i + 2](x)
        x = x.view(x.size(0), -1)
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)
            x = self.activation_fn(x)
            x = self.fc_layers[i + 1](x)
        x = self.output_layer(x)

        x = x.view(x.size(0), 100, 100)
        return x

class CNNBModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNBModel, self).__init__()

        self.output_shape = output_shape
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        self.flatten_size = self._compute_flatten_size(input_shape)

        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        self.output_layer = nn.Linear(input_fc_dim, int(np.prod(output_shape)))

        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)
            x = self.activation_fn(x)
            x = self.cnn_layers[i + 1](x)
            x = self.cnn_layers[i + 2](x)
        x = x.view(x.size(0), -1)
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)
            x = self.activation_fn(x)
            x = self.fc_layers[i + 1](x)

        x = self.output_layer(x)
        x = x.view(-1, *self.output_shape)

        return x

input_shape1 = (2, nfar, nfar)
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

CNNB_opt.load_state_dict(torch.load('Scattering/CNNB_tuned_model_kvalue1.pt'))
BCNN_opt.load_state_dict(torch.load('Scattering/BCNN_tuned_model_kvalue1.pt'))
CNN_opt.load_state_dict(torch.load('Scattering/CNN_tuned_model_kvalue1.pt'))

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
CNNB_opt_preds = np.concatenate(CNNB_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, 2, nfar, nfar)

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


# Make grid for NIO's DeepONet
grid_x = np.tile(np.linspace(0, 1, 100), (100, 1))
grid_y = np.tile(np.linspace(0, 1, 100), (100, 1)).T
grid = torch.tensor(np.stack([grid_y, grid_x], axis=-1)).type(torch.float32).to(device)

NIO_opt = torch.load("ModelSelection_final_s_born_farfield_nio/Setup_13/model.pkl", map_location=device, weights_only=False)
NIO_opt_preds = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of parameters for CNNB model: {count_parameters(CNNB_opt)}")
print(f"Number of trainable parameters for CNNB model: {count_trainable_parameters(CNNB_opt)}")
print(f"Number of parameters for CNNB model {count_parameters(BCNN_opt)}")
print(f"Number of trainable parameters for CNNB model: {count_trainable_parameters(BCNN_opt)}")
print(f"Number of parameters for CNN model: {count_parameters(CNN_opt)}")
print(f"Number of trainable parameters for CNN model: {count_trainable_parameters(CNN_opt)}")
print(f"Number of parameters for NIO model: {count_parameters(NIO_opt)}")
print(f"Number of trainable parameters for NIO model: {count_trainable_parameters(NIO_opt)}")

# NIO preprocesses data with minmax normalization on the real and imaginary parts of far field (or whatever you used to train it)
# Hack but I'm lazy
dataNIO = torch.load("Scattering/farfield_image_test.pt", map_location=device)
farfieldNIO = dataNIO["farfield"].to(device)
farfield_imag, farfield_real = farfieldNIO[:, 1], farfieldNIO[:, 0]
min_data_real = torch.min(farfield_real)
max_data_real = torch.max(farfield_real)
min_data_imag = torch.min(farfield_imag)
max_data_imag = torch.max(farfield_imag)
farfield_real = 2 * (farfield_real - min_data_real) / (max_data_real - min_data_real) - 1.
farfield_imag = 2 * (farfield_imag - min_data_imag) / (max_data_imag - min_data_imag) - 1.
farfieldNIO = torch.stack([farfield_real, farfield_imag], dim=-1)

# print(farfieldNIO.shape)
farfieldNIO = farfieldNIO.view(-1, 2, 100, 100)  # Reshape to (N_TEST_SAMPLES, 2, 100, 100)
farfieldNIO = farfieldNIO.to(device)
farfieldNIO = (1+torch.randn(farfieldNIO.shape).to(device)*delta_noise)*farfieldNIO  # Add mult noise
test_loader_NIO = DataLoader(TensorDataset(farfieldNIO), batch_size=32, shuffle=False)

max_output = torch.max(images)
min_output = torch.min(images)

print("NIO Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader_NIO, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = NIO_opt(inputs, grid)
        # unnormalize
        outputs = 0.5 * (outputs + 1) * (max_output - min_output) + min_output
        NIO_opt_preds.append(outputs.cpu().numpy())

NIO_opt_preds = np.concatenate(NIO_opt_preds, axis=0)  # shape: (N_TEST_SAMPLES, 2, nfar, nfar)

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

l2error = {"Born1": [], "NIO": [], "CNNB": [], "BCNN": [], "CNN": []}
l1error = {"Born1": [], "NIO": [], "CNNB": [], "BCNN": [], "CNN": []}

N_TEST_SAMPLES = 40
for i in range(N_TEST_SAMPLES):
    print("Sample: ", i)
    ground_truth = images[i].cpu().numpy().flatten()
    true_farfield =  farfields[i].reshape(nfar*nfar,1)

    NIO_opt_pred = NIO_opt_preds[i]
    CNNB_opt_pred = (CNNB_opt_preds[i][0]+1J*CNNB_opt_preds[i][1]).reshape(nfar*nfar,1)
    BCNN_opt_pred = BCNN_opt_preds[i].reshape(Ngrid,Ngrid)
    CNN_opt_pred = CNN_opt_preds[i].reshape(Ngrid,Ngrid)

    print("Computing Born")
    born1 = np.real(lsqr(born, true_farfield, damp=1e0)[0]).flatten()

    print("Computing NNs")
    nio = NIO_opt_pred.flatten()
    nn1 = np.real(lsqr(born, CNNB_opt_pred, damp=1e-1)[0]).flatten()
    nn2 = BCNN_opt_pred.flatten()+born1
    nn3 = CNN_opt_pred.flatten()

    print("Computing Norms")
    l2gt = np.linalg.norm(ground_truth, ord=2)
    print(l2gt)

    l2error["Born1"].append(np.linalg.norm(ground_truth - born1, ord=2)/l2gt)
    l2error["NIO"].append(np.linalg.norm(ground_truth - nio, ord=2)/l2gt)
    l2error["CNNB"].append(np.linalg.norm(ground_truth - nn1, ord=2)/l2gt)
    l2error["BCNN"].append(np.linalg.norm(ground_truth - nn2, ord=2)/l2gt)
    l2error["CNN"].append(np.linalg.norm(ground_truth - nn3, ord=2)/l2gt)

    l1gt = np.linalg.norm(ground_truth, ord=1)
    l1error["Born1"].append(np.linalg.norm(ground_truth - born1, ord=1)/l1gt)
    l1error["NIO"].append(np.linalg.norm(ground_truth - nio, ord=1)/l1gt)
    l1error["CNNB"].append(np.linalg.norm(ground_truth - nn1, ord=1)/l1gt)
    l1error["BCNN"].append(np.linalg.norm(ground_truth - nn2, ord=1)/l1gt)
    l1error["CNN"].append(np.linalg.norm(ground_truth - nn3, ord=1)/l1gt)

    nio = nio.reshape(100, 100)
    nn1 = nn1.reshape(100, 100)
    nn2 = nn2.reshape(100, 100)
    nn3 = nn3.reshape(100, 100)
    born1 = born1.reshape(100, 100)
    ground_truth = ground_truth.reshape(100, 100)

    baseline = [ground_truth+1, born1+1, nio+1]
    baseline_titles = ["(I.) Ground Truth", r"(II.) Born ($\gamma=1$)", r"(III.) NIO"]
    nndata = [nn1+1,nn2+1,nn3+1]
    nndata_titles = ["(IV.) CNNB", "(V.) BCNN", "(VI.) CNN"]

    vmin = .8
    vmax = 2
    print(np.real(ground_truth).max())

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=False)
    for j in range(3):
        im1 = axes[0, j].imshow(baseline[j], origin="lower", cmap='cmo.dense', vmin=vmin, vmax=vmax)
        axes[0, j].set_xlabel(baseline_titles[j], fontsize=12)
        if baseline_titles[j] == "(I.) Ground Truth":
            ticks = [0, 50, 99]
        else:
            ticks = [0, 50, 99]
        axes[0, j].set_xticks(ticks)
        axes[0, j].set_xticklabels(tick_labels)
        axes[0, j].set_yticks(ticks)
        axes[0, j].set_yticklabels(tick_labels)

        ticks = [0, 50, 99]
        im2 = axes[1, j].imshow(nndata[j], origin="lower", cmap='cmo.dense', vmin=vmin, vmax=vmax)
        axes[1, j].set_xlabel(nndata_titles[j], fontsize=12)
        axes[1, j].set_xticks(ticks)
        axes[1, j].set_xticklabels(tick_labels)
        axes[1, j].set_yticks(ticks)
        axes[1, j].set_yticklabels(tick_labels)
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    plt.savefig(folder / f'test_result_{i}.png', bbox_inches='tight')
    plt.close(fig)
l2error["BCNN_mean"] = np.mean(l2error["BCNN"])
l1error["BCNN_mean"] = np.mean(l1error["BCNN"])
l2error["CNNB_mean"] = np.mean(l2error["CNNB"])
l1error["CNNB_mean"] = np.mean(l1error["CNNB"])
l2error["NIO_mean"] = np.mean(l2error["NIO"])
l1error["NIO_mean"] = np.mean(l1error["NIO"])
l2error["Born1_mean"] = np.mean(l2error["Born1"])
l1error["Born1_mean"] = np.mean(l1error["Born1"])
l2error["CNN_mean"] = np.mean(l2error["CNN"])
l1error["CNN_mean"] = np.mean(l1error["CNN"])
dfl2 = pd.DataFrame(l2error)
dfl1 = pd.DataFrame(l1error)

print(dfl2)
print(dfl1)

dfl2.to_csv("Scattering/637_l2.csv", index=False)
dfl1.to_csv("Scattering/637_l1.csv", index=False)

