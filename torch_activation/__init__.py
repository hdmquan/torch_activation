from .other import Phish, DELU
from .layer import ReGLU, GeGLU, SeGLU, SwiGLU, LinComb, NormLinComb
from .radial import ScaledSoftSign
from .ridge import ShiLU, CReLU, ReLUN, SquaredReLU, StarReLU, CoLU
from .periodic import SinLU, CosLU, GCU
from .utils import plot_activation


__all__ = [
    "ShiLU",
    "DELU",
    "CReLU",
    "GCU",
    "CosLU",
    "CoLU",
    "ReLUN",
    "SquaredReLU",
    "ScaledSoftSign",
    "ReGLU",
    "GeGLU",
    "SeGLU",
    "SwiGLU",
    "LinComb",
    "NormLinComb",
    "SinLU",
    "Phish",
    "StarReLU",
    "plot_activation",
]

__version__ = "1.0.0"
