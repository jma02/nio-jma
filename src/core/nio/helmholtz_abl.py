from core.deeponet.DeepONetModules import DeepOnetNoBiasOrg, FeedForwardNN
from core.fno.FNOModules import FNO_WOR
from utils.Baselines import EncoderHelm2


import torch
import torch.nn as nn


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
        # self.fno_inputs = fno_input_dimension
        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)
        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderHelm2(output_dimensions)
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

        # x has shape N x L x nb
        L = x.shape[1]

        nx = (grid.shape[0])
        ny = (grid.shape[1])

        grid_r = grid.view(-1, 2)
        x = self.deeponet(x, grid_r)
        # x = self.attention(x)

        # x = x.view(x.shape[0], x.shape[1], nx, ny)

        # x = x.reshape(x.shape[0], x.shape[1], nx * ny)

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
        # print("Attention prams:")
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