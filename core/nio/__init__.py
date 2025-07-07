"""
Neural Inverse Operators (NIO) module

This module provides various implementations of Neural Inverse Operators for different problem domains.
Classes are organized by problem type and include both main implementations and ablation studies.
"""

# Helmholtz classes
from .helmholtz.helmholtz_nio import SNOHelmConv, NIOHelmPermInv
from .helmholtz.helmholtz_ablation import NIOHelmPermInvAbl

# Radiative classes
from .radiative.radiative_nio import SNOConvRad, NIORadPerm
from .radiative.radiative_ablation import NIORadPermAbl

# Wave classes
from .wave.wave_nio import SNOWaveConv2, NIOWavePerm
from .wave.wave_ablation import NIOWavePermAbl

# Medical classes
from .medical.medical_nio import NIOHeartPerm, SNOConvEIT
from .medical.medical_ablation import NIOHeartPermAbl

__all__ = [
    # Helmholtz
    'SNOHelmConv', 'NIOHelmPermInv', 'NIOHelmPermInvAbl',
    # Radiative
    'SNOConvRad', 'NIORadPerm', 'NIORadPermAbl',
    # Wave
    'SNOWaveConv2', 'NIOWavePerm', 'NIOWavePermAbl',
    # Medical
    'NIOHeartPerm', 'SNOConvEIT', 'NIOHeartPermAbl'
]