"""
Photo-Gen: A diffusion model for photonic structure generation.

This package provides tools for training and inference with diffusion models
specifically designed for generating photonic structures.
"""

__version__ = "0.1.0"
__author__ = "Vincent Letourneau"
__email__ = "poutine-dejeuner@github.com"

from .models import *
from .utils import *
from .evaluation import *

__all__ = [
    # Models
    "UNET",
    "ResBlock", 
    "Attention",
    "UnetLayer",
    "SinusoidalEmbeddings",
    
    # Utils
    "DDPM_Scheduler",
    "set_seed",
    "UNetPad",
    "display_reverse",
    
    # Evaluation
    "evaluation",
    "compute_FOM_parallele",
]