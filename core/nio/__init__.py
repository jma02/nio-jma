"""
Neural Inverse Operators (NIO) module

This module provides various implementations of Neural Inverse Operators for different problem domains.
Classes are organized by problem type and include both main implementations and ablation studies.
"""

# Import key components from submodules with error handling
try:
    from .helmholtz import (
        SNOHelmConv,
        NIOHelmPermInv,
        NIOHelmPermInvAbl
    )
    HELMHOLTZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import helmholtz module: {e}")
    HELMHOLTZ_AVAILABLE = False

try:
    from .radiative import (
        SNOConvRad,
        NIORadPerm,
        NIORadPermAbl
    )
    RADIATIVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import radiative module: {e}")
    RADIATIVE_AVAILABLE = False

from .wave import (
    SNOWaveConv2,
    NIOWavePerm,
    NIOWavePermAbl
)

from .medical import (
    NIOHeartPerm,
    NIOHeartPermAbl,
    SNOConvEIT
)

# Re-export key components
__all__ = [
    # Helmholtz models
    'SNOHelmConv',
    'NIOHelmPermInv',
    'NIOHelmPermInvAbl',
    'HELMHOLTZ_AVAILABLE',
    
    # Radiative models
    'SNOConvRad',
    'NIORadPerm',
    'NIORadPermAbl',
    'RADIATIVE_AVAILABLE',
    
    # Wave models
    'SNOWaveConv2',
    'NIOWavePerm',
    'NIOWavePermAbl',
    
    # Medical models
    'NIOHeartPerm',
    'NIOHeartPermAbl',
    'SNOConvEIT',
    
    # Submodules
    'helmholtz',
    'radiative',
    'wave',
    'medical'
]