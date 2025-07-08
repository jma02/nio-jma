import numpy as np
import torch
import torch.nn as nn

from core.deeponet import FeedForwardNN, DeepOnetNoBiasOrg
from core.fno import FNO2d, FNO_WOR
from utils.Baselines import EncoderRad, EncoderRad2

class NIOHeartPerm(nn.Module):
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
        super(NIOHeartPerm, self).__init__()
        output_dimensions = network_properties_trunk["n_basis"]
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed

        self.fno_inputs = fno_input_dimension
        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)
        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderRad2(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(2 + 1, fno_architecture["width"])
        if self.fno_layers != 0:
            self.fno = FNO_WOR(fno_architecture, device=device, padding_frac=padding_frac)
        self.device = device

    def forward(self, x, grid):
        nx = (grid.shape[0])
        ny = (grid.shape[1])
        x = self.deeponet(x, grid)
        x = x.reshape(-1, nx, ny, 1)

        grid = grid.unsqueeze(0)
        grid = grid.expand(x.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
        x = torch.cat((x, grid), dim=-1)
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


class SNOConvEIT(nn.Module):
    # I don't know why the S is there, I think it's supposed to be an F? - Jonathan
    def __init__(self,
                 input_dimensions_branch,
                 input_dimensions_trunk,
                 network_properties_branch,
                 network_properties_trunk,
                 fno_architecture,
                 device,
                 retrain_seed):
        super(SNOConvEIT, self).__init__()
        output_dimensions = network_properties_trunk["n_basis"]
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed

        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)
        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderRad(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)

        if self.fno_layers != 0:
            self.fno = FNO2d(fno_architecture)

        self.device = device

    def forward(self, x, grid):
        nx = (grid.shape[0])
        ny = (grid.shape[1])
        dim = (grid.shape[2])
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
