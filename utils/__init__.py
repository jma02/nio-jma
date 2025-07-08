"""
NIO-JMA Utilities

This package provides utility functions and classes for the NIO-JMA project.
"""

# Import key components to make them available at the package level
from .Baselines import (
    EncoderHelm, 
    EncoderHelm2, 
    EncoderRad, 
    EncoderRad2, 
    EncoderInversionNet,
    EncoderInversionNet2,
    InversionNet,
    InversionNetHelm,
    InversionNetRad,
    InversionNetEIT
)
from .debug_tools import *
from .transforms import *
from .SolveHelmTorch import solve_helm
# from .ConstOpt import *

# Re-export key components
__all__ = [
    # Encoders
    'EncoderHelm',
    'EncoderHelm2',
    'EncoderRad',
    'EncoderRad2',
    'EncoderInversionNet',
    'EncoderInversionNet2',
    
    # Models
    'InversionNet',
    'InversionNetHelm',
    'InversionNetRad',
    'InversionNetEIT',
    
    # Functions
    'solve_helm',
    
    # Modules
    'debug_tools',
    'transforms',
]
