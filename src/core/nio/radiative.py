import numpy as np
import torch
import torch.nn as nn

from utils.Baselines import EncoderRad, EncoderRad2
from core.deeponet.DeepONetModules import FeedForwardNN, \
    DeepOnetNoBiasOrg
from core.fno.FNOModules import FNO1d, FNO1d_WOR


class SNOConvRad(nn.Module):
    def __init__(self,
                 input_dimensions_branch,
                 input_dimensions_trunk,
                 network_properties_branch,
                 network_properties_trunk,
                 fno_architecture,
                 device,
                 retrain_seed):
        super(SNOConvRad, self).__init__()
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
            self.fno = FNO1d(fno_architecture)

        self.device = device

    def forward(self, x, grid):
        nx = (grid.shape[0])
        dim = 1
        grid_deeponet = grid.reshape(nx, dim)
        x = self.deeponet(x, grid_deeponet)
        x = x.reshape(-1, nx, 1)

        if self.fno_layers != 0:
            grid = grid.expand(x.shape[0], grid.shape[0]).unsqueeze(-1)
            x = torch.cat((x, grid), dim=-1)
            h = self.fno(x)
        else:
            h = x.squeeze(-1)
        return h

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



class NIORadPerm(nn.Module):
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
        super(NIORadPerm, self).__init__()
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
        self.fc0 = nn.Linear(1 + 1, fno_architecture["width"])

        if self.fno_layers != 0:
            self.fno = FNO1d_WOR(fno_architecture, device=device, padding_frac=padding_frac)

        self.device = device

    def forward(self, x, grid):
        x = x.unsqueeze(2)
        # x has shape N x L x nb
        if self.training:
            L = np.random.randint(2, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]
        # x has shape N x L x nb
        nx = (grid.shape[0])

        grid_r = grid.reshape(-1, 1)

        x = self.deeponet(x, grid_r)
        x = x.view(x.shape[0], x.shape[1], nx)
        grid = grid.unsqueeze(0)
        grid = grid.unsqueeze(1)
        grid = grid.expand(x.shape[0], 1, grid.shape[2])

        x = torch.cat((grid, x), 1)

        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        weight_trans_mat = torch.cat([weight_trans_mat[:, :1], weight_trans_mat[:, 1].view(-1, 1).repeat(1, L) / L], dim=1)

        x = x.permute(0, 2, 1)
        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat
        if self.fno_layers != 0:
            x = self.fno(x)
        else:
            x = x[..., 0]
        return x

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

