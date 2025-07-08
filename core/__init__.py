"""
NIO-JMA Core Package

This package contains the core functionality for Neural Inverse Operators.
"""

__version__ = '0.1.0'

# Import key components to make them available at the package level
from .deeponet import FeedForwardNN, DeepOnetNoBiasOrg, FourierFeatures
from .fno import FNO2d, FNO_WOR, FNO1d, FNO1d_WOR

# Import from nio subpackage
from .nio import *

# Re-export key components
__all__ = [
    # DeepONet components
    'FeedForwardNN',
    'DeepOnetNoBiasOrg',
    'FourierFeatures',
    
    # FNO components
    'FNO2d',
    'FNO_WOR',
    'FNO1d',
    'FNO1d_WOR',
    
    # Subpackages
    'deeponet',
    'fno',
    'nio'
]
