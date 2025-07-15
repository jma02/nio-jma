import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BornFarFieldDataset(Dataset):
    def __init__(self, norm, inputs_bool, device, which, mod, noise=0, samples=4096):
        print("Training with 20000 samples")
        self.file_data = "data/merged_data_split.hd5"
        self.mod = mod
        self.noise = noise
        self.which = which
        self.reader = h5py.File(self.file_data, 'r')
        print(self.file_data, self.reader.keys())

        self.mean_inp_real = torch.from_numpy(self.reader['mean_inp_fun_real'][:, :]).type(torch.float32)
        self.mean_inp_imag = torch.from_numpy(self.reader['mean_inp_fun_imag'][:, :]).type(torch.float32)
        self.mean_out = torch.from_numpy(self.reader['mean_out_fun'][:, :]).type(torch.float32)

        self.std_inp_real = torch.from_numpy(self.reader['std_inp_fun_real'][:, :]).type(torch.float32)
        self.std_inp_imag = torch.from_numpy(self.reader['std_inp_fun_imag'][:, :]).type(torch.float32)
        self.std_out = torch.from_numpy(self.reader['std_out_fun'][:, :]).type(torch.float32)

        self.min_data_real = torch.from_numpy(self.reader['min_inp_real'][:]).type(torch.float32)
        self.max_data_real = torch.from_numpy(self.reader['max_inp_real'][:]).type(torch.float32)
        self.min_data_imag = torch.from_numpy(self.reader['min_inp_imag'][:]).type(torch.float32)
        self.max_data_imag = torch.from_numpy(self.reader['max_inp_imag'][:]).type(torch.float32)

        self.min_model = torch.from_numpy(self.reader['min_out'][:]).type(torch.float32)
        self.max_model = torch.from_numpy(self.reader['max_out'][:]).type(torch.float32)

        self.inp_dim_branch = 2
        #self.n_fun_samples = 100

        self.norm = norm
        self.inputs_bool = inputs_bool

        self.device = device

        self.min_data_real_logt = torch.log(self.min_data_real + 1e-16)
        self.max_data_real_logt = torch.log(self.max_data_real + 1e-16)
        self.min_data_imag_logt = torch.log(self.min_data_imag + 1e-16)
        self.max_data_imag_logt = torch.log(self.max_data_imag + 1e-16)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Get input_real and input_imag from the group
        input_real = torch.from_numpy(self.reader[self.which]["input_real"][index]).type(torch.float32)
        input_imag = torch.from_numpy(self.reader[self.which]["input_imag"][index]).type(torch.float32)

        inputs_real = inputs_real * (1 + self.noise * torch.randn_like(inputs_real))
        inputs_imag = inputs_imag * (1 + self.noise * torch.randn_like(inputs_imag))
        # Get output/labels
        labels = torch.from_numpy(self.reader[self.which]["output"][index]).type(torch.float32)
        if self.norm == "norm":
            inputs_real = self.normalize(inputs_real, self.mean_inp_real, self.std_inp_real)
            inputs_imag = self.normalize(inputs_imag, self.mean_inp_imag, self.std_inp_imag)
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "norm-inp":
            inputs_real = self.normalize(inputs_real, self.mean_inp_real, self.std_inp_real)
            inputs_imag = self.normalize(inputs_imag, self.mean_inp_imag, self.std_inp_imag)
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "log-minmax":
            inputs_real = (np.log1p(np.abs(inputs_real))) * np.sign(inputs_real)
            inputs_real = 2 * (inputs_real - self.min_data_real_logt) / (self.max_data_real_logt - self.min_data_real_logt) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "norm-out":
            inputs_real = 2 * (inputs_real - self.min_data_real) / (self.max_data_real - self.min_data_real) - 1.
            inputs_imag = 2 * (inputs_imag - self.min_data_imag) / (self.max_data_imag - self.min_data_imag) - 1.
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "minmax":
            inputs_real = 2 * (inputs_real - self.min_data_real) / (self.max_data_real - self.min_data_real) - 1.
            inputs_imag = 2 * (inputs_imag - self.min_data_imag) / (self.max_data_imag - self.min_data_imag) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "none":
            pass
        else:
            raise ValueError()

        # Combine real and imaginary parts
        inputs = torch.stack([input_real, input_imag], dim=-1)

        inputs = inputs.view(2, 100, 100)

        return inputs, labels

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        if self.norm == "norm" or self.norm == "norm-out":
            return tensor * (self.std_out + 1e-16).to(self.device) + self.mean_out.to(self.device)
        elif self.norm == "none":
            return tensor
        else:
            return (self.max_model - self.min_model) * (tensor + torch.tensor(1., device=self.device)) / 2 + self.min_model.to(self.device)

    def get_grid(self):
        grid = torch.from_numpy(self.reader['grid'][:, :]).type(torch.float32)

        return grid.unsqueeze(0)
