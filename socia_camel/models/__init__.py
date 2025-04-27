"""
Models module for SOCIA-CAMEL system.
Contains predefined simulation models used by the system.
"""

from .epidemic_model import EpidemicModel
from .social_network_model import SocialNetworkModel

__all__ = [
    'EpidemicModel',
    'SocialNetworkModel'
]
