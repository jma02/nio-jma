import numpy as np
import torch
import torch.nn as nn

from core.fno import FNO2d, FNO_WOR
from core.deeponet import FeedForwardNN, DeepOnetNoBiasOrg
from utils.Baselines import EncoderHelm2

class NIOHelmPermInvAbl(nn.Module):
    def __init__(self,
                 input_dimensions_branch,
                 input_dimensions_trunk,
                 network_properties_branch,
                 network_properties_trunk,
                 fno_architecture,
                 device,
                 retrain_seed,
                 fno_input_dimension=1000,
                 padding_frac=1 / 4):
        super(NIOHelmPermInvAbl, self).__init__()
        output_dimensions = network_properties_trunk["n_basis"]
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed
        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)
        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderHelm2(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(3, fno_architecture["width"])
        if self.fno_layers != 0:
            self.fno = FNO_WOR(fno_architecture, device=device, padding_frac=padding_frac)
        self.device = device

    def forward(self, x, grid):
        if self.training:
            L = np.random.randint(2, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]

        nx = (grid.shape[0])
        ny = (grid.shape[1])
        grid_r = grid.view(-1, 2)
        x = self.deeponet(x, grid_r)
        x = x.view(x.shape[0], x.shape[1], nx, ny)

        grid = grid.unsqueeze(0)
        grid = grid.expand(x.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]).permute(0, 3, 1, 2)
        x = torch.cat((grid, x), 1)

        weight_trans_mat = self.fc0.weight.data
        weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = torch.matmul(x, weight_trans_mat.T) + self.fc0.bias.data
        if self.fno_layers != 0:
            x = self.fno(x)

        return x[:, :, :, 0]

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
