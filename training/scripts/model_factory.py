import sys
import os
import torch

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from core package
from core.nio import (
    # Wave models
    SNOWaveConv2, 
    NIOWavePerm, 
    NIOWavePermAbl,
    
    # Helmholtz models
    SNOHelmConv, 
    NIOHelmPermInv, 
    NIOHelmPermInvAbl,
    
    # Radiative models
    SNOConvRad, 
    NIORadPerm, 
    NIORadPermAbl,
    
    # Medical models
    NIOHeartPerm, 
    NIOHeartPermAbl, 
    SNOConvEIT
)

# Core components
from core.deeponet import FeedForwardNN, DeepOnetNoBiasOrg, FourierFeatures
from core.fno import FNO2d, FNO_WOR, FNO1d, FNO1d_WOR

class ModelFactory:
    @staticmethod
    def create_model(problem, mod, inp_dim_branch, grid_shape, branch_architecture, trunk_architecture, fno_architecture, device=None, retrain_seed=42):
        # Set default device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if mod == "nio" or mod == "don":
            if problem == "sine" or problem == "helm" or problem == "step":
                return SNOHelmConv(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
            elif problem == "rad":
                return SNOConvRad(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
            elif problem == "eit":
                return SNOConvEIT(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
        else:  # mod == "fcnio"
            if problem == "sine" or problem == "helm" or problem == "step":
                return NIOHelmPermInv(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
            elif problem == "rad":
                return NIORadPerm(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
            elif problem == "eit":
                return NIOHeartPerm(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
            elif problem == "wave":
                return NIOWavePerm(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture,
                    device=device,
                    retrain_seed=retrain_seed
                )
        raise ValueError(f"Unknown problem type: {problem}")
