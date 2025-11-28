from .wrapper import render

from .tests_wrapper import (
    barycentric,
    interpolate3_scalar,
    interpolate3_vector,
)
from .util import precision

__all__ = [
    "render",
    "barycentric",
    "interpolate3_scalar",
    "interpolate3_vector",
]

__version__ = "0003-no_print"
