"""Deployment-oriented AutoML. See docs/README.md for the full documentation."""

from .config import Config
from .model import FittedModel
from .trainer import AutoML

__version__ = "0.1.0"
__all__ = ["AutoML", "Config", "FittedModel", "__version__"]
