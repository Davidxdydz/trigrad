from .wrapper import render, precision

from .tests_wrapper import (
    barycentric,
    interpolate3_scalar,
    interpolate3_vector,
)


__all__ = [
    "render",
    "barycentric",
    "interpolate3_scalar",
    "interpolate3_vector",
]

__version__ = "0003-no_print"
