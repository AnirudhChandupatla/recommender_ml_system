"""
Recommender System Package
A production-ready ML system for product recommendations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import config
from . import etl
from . import models
from . import api
from . import utils

__all__ = [
    'config',
    'etl',
    'models',
    'api',
    'utils',
]
