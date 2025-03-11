"""Module with the classes for the projected normal distribution."""
from .general_projnormal import ProjNormal as ProjNormal
from .const_projnormal import ProjNormalConst as ProjNormalConst

__all__ = [
  "ProjNormal",
  "ProjNormalConst"
]

def __dir__():
    return __all__
