"""Module with the classes for the projected normal distribution."""
from .general_projected_normal import ProjNormal as ProjNormal
from .c50_projected_normal import ProjNormalConst as ProjNormalConst

__all__ = [
  "ProjNormal",
  "ProjNormalConst"
]

def __dir__():
    return __all__
