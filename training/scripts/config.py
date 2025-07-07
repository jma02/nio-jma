import json
import sys
import os

class Config:
    def __init__(self, folder, training_properties, branch_architecture, trunk_architecture, fno_architecture, denseblock_architecture, problem, mod, max_workers):
        self.folder = folder
        self.training_properties = training_properties
        self.branch_architecture = branch_architecture
        self.trunk_architecture = trunk_architecture
        self.fno_architecture = fno_architecture
        self.denseblock_architecture = denseblock_architecture
        self.problem = problem
        self.mod = mod
        self.max_workers = max_workers

    @classmethod
    def from_args(cls, folder, *args):
        if len(args) == 4:
            training_properties = {
                "step_size": 15,
                "gamma": 1,
                "epochs": 100,
                "batch_size": 256,
                "learning_rate": 0.001,
                "norm": "log-minmax",
                "weight_decay": 0,
                "reg_param": 0,
                "reg_exponent": 1,
                "inputs": 2,
                "b_scale": 0.,
                "retrain": 888,
                "mapping_size_ff": 32,
                "scheduler": "step"
            }
            branch_architecture = {
                "n_hidden_layers": 3,
                "neurons": 64,
                "act_string": "leaky_relu",
                "dropout_rate": 0.0,
                "kernel_size": 3
            }
            trunk_architecture = {
                "n_hidden_layers": 8,
                "neurons": 256,
                "act_string": "leaky_relu",
                "dropout_rate": 0.0,
                "n_basis": 50
            }
            fno_architecture = {
                "width": 64,
                "modes": 16,
                "n_layers": 1,
            }
            denseblock_architecture = {
                "n_hidden_layers": 4,
                "neurons": 2000,
                "act_string": "leaky_relu",
                "retrain": 56,
                "dropout_rate": 0.0
            }
            problem = args[1]
            mod = args[2]
            max_workers = int(args[3])
        else:
            training_properties = json.loads(args[1].replace("\'", '"'))
            branch_architecture = json.loads(args[2].replace("\'", '"'))
            trunk_architecture = json.loads(args[3].replace("\'", '"'))
            fno_architecture = json.loads(args[4].replace("\'", '"'))
            denseblock_architecture = json.loads(args[5].replace("\'", '"'))
            problem = args[6]
            mod = args[7]
            max_workers = int(args[8])
        
        return cls(folder, training_properties, branch_architecture, trunk_architecture, fno_architecture, denseblock_architecture, problem, mod, max_workers)

    def save_config(self):
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
            
            df = pd.DataFrame.from_dict([self.training_properties]).T
            df.to_csv(os.path.join(self.folder, 'training_properties.txt'), header=False, index=True, mode='a')
            
            df = pd.DataFrame.from_dict([self.branch_architecture]).T
            df.to_csv(os.path.join(self.folder, 'branch_architecture.txt'), header=False, index=True, mode='a')
            
            df = pd.DataFrame.from_dict([self.trunk_architecture]).T
            df.to_csv(os.path.join(self.folder, 'trunk_architecture.txt'), header=False, index=True, mode='a')
            
            df = pd.DataFrame.from_dict([self.fno_architecture]).T
            df.to_csv(os.path.join(self.folder, 'fno_architecture.txt'), header=False, index=True, mode='a')
            
            df = pd.DataFrame.from_dict([self.denseblock_architecture]).T
            df.to_csv(os.path.join(self.folder, 'denseblock_architecture.txt'), header=False, index=True, mode='a')

    def get_problem_class(self):
        if self.problem == "sine":
            from Problems.PoissonSin import PoissonSinDataset
            return PoissonSinDataset
        elif self.problem == "helm":
            from Problems.helmholtz.HelmNIO import HelmNIOData
            return HelmNIOData
        elif self.problem == "curve":
            from Problems.curve_fwi.CurveVel import CurveVelDataset
            return CurveVelDataset
        elif self.problem == "style":
            from Problems.StyleData import StyleData
            return StyleData
        elif self.problem == "rad":
            from Problems.AlbedoOperator import AlbedoData
            return AlbedoData
        elif self.problem == "eit":
            from Problems.medical.HeartLungsEIT import HeartLungsEITDataset
            return HeartLungsEITDataset
        else:
            raise ValueError(f"Unknown problem type: {self.problem}")
