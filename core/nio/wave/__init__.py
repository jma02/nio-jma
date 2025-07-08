"""
Wave module for Neural Inverse Operators.
"""

from .wave_nio import SNOWaveConv2
from .wave_ablation import NIOWavePerm, NIOWavePermAbl

__all__ = [
    'SNOWaveConv2',
    'NIOWavePerm',
    'NIOWavePermAbl'
]