# Dual-Branch VAE Project
# Pure Python implementation for SVG processing and VAE training

__version__ = "1.0.0"
__author__ = "Text2SVG Project"

# Make key modules easily accessible
from . import models
from . import data
from . import utils
from . import losses
from . import pydiffvg_lite

__all__ = [
    'models',
    'data', 
    'utils',
    'losses',
    'pydiffvg_lite'
]
