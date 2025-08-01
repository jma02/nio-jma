import numpy as np
import torch
import torch.nn as nn

from utils.Baselines import EncoderInversionNet, EncoderInversionNet2
from core.deeponet.DeepONetModules import FeedForwardNN, \
    FourierFeatures, DeepOnetNoBiasOrg
from core.fno.FNOModules import FNO2d, FNO_WOR

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

class NIOWavePerm(nn.Module):
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
        super(NIOWavePerm, self).__init__()
        output_dimensions = network_properties_trunk["n_basis"]
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed

        self.fno_inputs = fno_input_dimension
        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)

        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderInversionNet2(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        # self.fc0 = nn.Linear(2 + 2, fno_architecture["width"])
        # self.fc0 = nn.Linear(2 + 1, fno_architecture["width"])
        self.fc0 = nn.Linear(3, fno_architecture["width"])
        # self.correlation_network = nn.Sequential(nn.Linear(2, 50), nn.LeakyReLU(),
        #                                         nn.Linear(50, 50), nn.LeakyReLU(),
        #                                         nn.Linear(50, 1)).to(device)
        if self.fno_layers != 0:
            self.fno = FNO_WOR(fno_architecture, device=device, padding_frac=padding_frac)
        # self.attention = Attention(70 * 70, res=70 * 70)
        self.device = device

    def forward(self, x, grid):

        if self.training:
            L = np.random.randint(2, x.shape[3])
            idx = np.random.choice(x.shape[3], L)
            x = x[:, :, :, idx]
        else:
            L = x.shape[3]

        # x has shape N x L x nb
        nx = (grid.shape[0])
        ny = (grid.shape[1])
        grid_r = grid.reshape(-1, 2)

        # x_i = x_i.permute(1, 0)
        x = self.deeponet(x, grid_r)
        # x = self.attention(x)
        x = x.view(x.shape[0], x.shape[1], nx, ny)
        grid = grid.unsqueeze(0)
        grid = grid.expand(x.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]).permute(0, 3, 1, 2)
        x = torch.cat((grid, x), 1)
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        # weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L), weight_trans_mat[:, 3].view(-1, 1)], dim=1)
        weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)
        # weight_trans_mat = torch.cat([weight_trans_mat.repeat(1, L)], dim=1)
        x = x.permute(0, 2, 3, 1)
        # input_corr = x[..., np.random.randint(0, L, 2)]
        # out_corr = self.correlation_network(input_corr)
        # x = torch.concat((x, out_corr), -1)
        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat
        h = self.fno(x)
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
        print("Attention prams:")
        # a_size = self.attention.print_size()

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
