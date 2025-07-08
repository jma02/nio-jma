import numpy as np
import torch
import torch.nn as nn

from core.deeponet import FeedForwardNN, DeepOnetNoBiasOrg, FourierFeatures
from core.fno import FNO2d, FNO_WOR
from utils.Baselines import EncoderInversionNet

class SNOWaveConv2(nn.Module):
    def __init__(self,
                 input_dimensions_branch,
                 input_dimensions_trunk,
                 network_properties_branch,
                 network_properties_trunk,
                 fno_architecture,
                 device,
                 retrain_seed,
                 b_scale,
                 mapping_size):
        super(SNOWaveConv2, self).__init__()
        output_dimensions = network_properties_trunk["n_basis"]
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed
        if b_scale != 0.0:
            self.trunk = FeedForwardNN(2 * mapping_size, output_dimensions, network_properties_trunk)
            self.fourier_features_transform = FourierFeatures(b_scale, mapping_size, device)
        else:
            self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)

        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderInversionNet(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)

        if self.fno_layers != 0:
            self.fno = FNO2d(fno_architecture, device=device)

        self.device = device
        self.b_scale = b_scale

    def forward(self, x, grid):
        nx = (grid.shape[0])
        ny = (grid.shape[1])
        dim = (grid.shape[2])
        if self.b_scale != 0.0:
            grid_deeponet = self.fourier_features_transform(grid)
        else:
            grid_deeponet = grid.reshape(-1, dim)
        x = self.deeponet(x, grid_deeponet)
        x = x.reshape(-1, nx, ny, 1)

        if self.fno_layers != 0:
            grid = grid.unsqueeze(0)
            grid = grid.expand(x.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
            x = torch.cat((x, grid), dim=-1)
            h = self.fno(x)
        else:
            h = x
        return h[:, :, :, 0]

    def print_size(self):
        print("Branch prams:")
        b_size = self.branch.print_size()
        print("Trunk prams:")
        t_size = self.trunk.print_size()
        if self.fno_layers != 0:
            print("FNO prams:")
            f_size = self.fno.print_size()
        else:
            print("NO FNO")

        if self.fno_layers != 0:
            size = b_size + t_size + f_size
        else:
            size = b_size + t_size
        print(size)
        return size

    def regularization(self, q):
        reg_loss = 0
        for name, param in self.named_parameters():
            reg_loss = reg_loss + torch.norm(param, q)
        return reg_loss
