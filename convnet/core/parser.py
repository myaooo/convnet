"""
parser.py

Helpers that deal with model loading and freezing etc.
"""

from convnet.core import ConvNet
from convnet.core.layers import InputLayer



def layercode(layer):
    return str(layer.__class__.__name__).lower()[:-5]
