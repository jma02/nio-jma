from core.nio.eit import NIOHeartPerm, SNOConvEIT
from core.nio.helmholtz import NIOHelmPermInv, SNOHelmConv
from core.nio.radiative import NIORadPerm, SNOConvRad
from core.nio.wave import NIOWavePerm, SNOWaveConv2
from utils.Baselines import InversionNetEIT, InversionNetHelm, InversionNetRad


def get_training_properties():
    return  {
        "step_size": 15,
        "gamma": 1,
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.01,
        "norm": "minmax",
        "weight_decay": 0,
        "reg_param": 0,
        "reg_exponent": 1,
        "inputs": 2,
        "b_scale": 0.,
        "retrain": 888,
        "mapping_size_ff": 32,
        "scheduler": "step"
    }

def get_branch_architecture():
    return {
        "n_hidden_layers": 3,
        "neurons": 64,
        "act_string": "leaky_relu",
        "dropout_rate": 0.0,
        "kernel_size": 3
    }

def get_trunk_architecture():
    return  {
        "n_hidden_layers": 8,
        "neurons": 256,
        "act_string": "leaky_relu",
        "dropout_rate": 0.0,
        "n_basis": 50
    }

def get_fno_architecture():
    return {
        "width": 64,
        "modes": 16,
        "n_layers": 1,
    }

def get_denseblock_architecture():
    return {
        "n_hidden_layers": 4,
        "neurons": 2000,
        "act_string": "leaky_relu",
        "retrain": 56,
        "dropout_rate": 0.0
    }
 

def get_model(
    mod,
    problem,
    inp_dim_branch,
    grid,
    branch_architecture_,
    trunk_architecture_,
    fno_architecture_,
    denseblock_architecture_,
    device,
    retrain_seed,
    fno_input_dimension,
    b_scale=None,
    mapping_size=None
):
    """
    Helper function to construct the model based on mod and problem.
    Returns the instantiated model.
    """
    if mod in ["nio", "don"]:
        print("Using CNIO")
        if problem in ["sine", "helm", "step", "born_farfield"]:
            return SNOHelmConv(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=grid.shape[2],
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed
            )
        elif problem in ["curve", "style"]:
            return SNOWaveConv2(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=grid.shape[2],
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed,
                b_scale=b_scale,
                mapping_size=mapping_size
            )
        elif problem == "rad":
            return SNOConvRad(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=1,
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed
            )
        elif problem == "eit":
            return SNOConvEIT(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=grid.shape[2],
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed
            )
    elif mod == "fcnn":
        print("Using FCNN")
        if problem in ["sine", "helm", "step"]:
            return InversionNetHelm(int(branch_architecture_["neurons"]))
        elif problem == "rad":
            return InversionNetRad(int(branch_architecture_["neurons"]))
        elif problem == "eit":
            return InversionNetEIT(int(branch_architecture_["neurons"]))
    else:
        print("Using FCNIO")
        if problem in ["sine", "helm", "step"]:
            return NIOHelmPermInv(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=grid.shape[2],
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed,
                fno_input_dimension=fno_input_dimension
            )
        elif problem in ["curve", "style"]:
            return NIOWavePerm(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=grid.shape[2],
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed,
                fno_input_dimension=fno_input_dimension
            )
        elif problem == "rad":
            return NIORadPerm(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=1,
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed,
                fno_input_dimension=fno_input_dimension
            )
        elif problem == "eit":
            return NIOHeartPerm(
                input_dimensions_branch=inp_dim_branch,
                input_dimensions_trunk=2,
                network_properties_branch=branch_architecture_,
                network_properties_trunk=trunk_architecture_,
                fno_architecture=fno_architecture_,
                device=device,
                retrain_seed=retrain_seed,
                fno_input_dimension=fno_input_dimension
            )