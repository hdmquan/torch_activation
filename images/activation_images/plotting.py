from torch_activation import *
from torch_activation.utils import plot_activation

# ReLUs
relun_p = {"n": [1, 3]}
shilu_p = {"alpha": [1, 2], "beta": [0, 1]}

plot_activation(ReLUN, relun_p)
plot_activation(ShiLU, shilu_p)
plot_activation(SquaredReLU)
plot_activation(StarReLU)
plot_activation(ShiftedReLU)
plot_activation(SlReLU)
# plot_activation(SoftsignRReLU)
plot_activation(SineReLU)
plot_activation(Minsin)
plot_activation(VLU)

# Piece-wise

delu_p = {"n": [1, 3]}

plot_activation(DELU, delu_p)


# Trig

coslu_p = {"a": [1, 2], "b": [1, 1]}
sinlu_p = {"a": [1, 2], "b": [1, 1]}

plot_activation(CosLU, coslu_p)
plot_activation(SinLU, sinlu_p)
plot_activation(GCU)


# Curves

s3_p = {"alpha": [1, 2], "beta": [1, 2]}

plot_activation(CoLU)
plot_activation(Phish)
plot_activation(ScaledSoftSign, s3_p, y_range=(-2, 2))


# LinComb

plot_activation(LinComb)
plot_activation(NormLinComb)
