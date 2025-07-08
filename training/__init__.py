"""
NIO-JMA Training Module

This module provides training utilities and scripts for Neural Inverse Operators.
"""

# Import key components to make them available at the package level
from .scripts.trainer import Trainer
from .scripts.config import Config
from .scripts.model_factory import ModelFactory
from .scripts.RunNio import main

# Re-export key components
__all__ = [
    'Trainer',
    'Config',
    'ModelFactory',
    'main',
    'scripts'
]
