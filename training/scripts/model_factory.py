from nio.helmholtz.helmholtz_nio import SNOHelmConv, NIOHelmPermInv
from nio.wave.wave_nio import SNOWaveConv2, SNOConvEIT, NIOWavePerm
from nio.radiative.radiative_nio import SNOConvRad, NIORadPerm
from nio.medical.medical_nio import NIOHeartPerm
import torch

class ModelFactory:
    @staticmethod
    def create_model(problem, mod, inp_dim_branch, grid_shape, branch_architecture, trunk_architecture, fno_architecture):
        if mod == "nio" or mod == "don":
            if problem == "sine" or problem == "helm" or problem == "step":
                return SNOHelmConv(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
            elif problem == "rad":
                return SNOConvRad(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
            elif problem == "eit":
                return SNOConvEIT(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
        else:  # mod == "fcnio"
            if problem == "sine" or problem == "helm" or problem == "step":
                return NIOHelmPermInv(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
            elif problem == "rad":
                return NIORadPerm(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
            elif problem == "eit":
                return NIOHeartPerm(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
            elif problem == "wave":
                return NIOWavePerm(
                    input_dimensions_branch=inp_dim_branch,
                    input_dimensions_trunk=grid_shape[2],
                    network_properties_branch=branch_architecture,
                    network_properties_trunk=trunk_architecture,
                    fno_architecture=fno_architecture
                )
        raise ValueError(f"Unknown problem type: {problem}")
