from .other import Phish, DELU
from .layer import ReGLU, GeGLU, SeGLU, SwiGLU, LinComb, NormLinComb
from .adaptive_relu import ShiLU, SquaredReLU, StarReLU, CoLU
from .relu import ReLUN, CReLU, SlReLU, ShiftedReLU, SoftsignRReLU
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
    "SlReLU",
    "SoftsignRReLU",
    "ShiftedReLU",
    "plot_activation",
]

__version__ = "1.0.0"
