"""Module with the classes for the projected normal distribution."""
from .general_projected_normal import ProjectedNormal as ProjectedNormal
from .c50_projected_normal import ProjectedNormalConst as ProjectedNormalConst

__all__ = [
  "ProjectedNormal",
  "ProjectedNormalConst"
]

def __dir__():
    return __all__
