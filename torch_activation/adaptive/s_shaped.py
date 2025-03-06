import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation

@register_activation
class SReLU(nn.Module):
    r"""
    Applies the S-shaped Rectified Linear Unit (SReLU) function:

    :math:`\text{SReLU}(z_i) = \begin{cases} 
    t^r_i + a^r_i(z_i - t^r_i), & z_i \geq t^r_i \\
    z_i, & t^r_i > z_i > t^l_i \\
    t^l_i + a^l_i(z_i - t^l_i), & z_i \leq t^l_i
    \end{cases}`

    where :math:`t^r_i`, :math:`t^l_i`, :math:`a^r_i`, and :math:`a^l_i` are trainable parameters.

    Args:
        init_tr (float, optional): Initial value for the right threshold parameter tr. Default: 1.0
        init_tl (float, optional): Initial value for the left threshold parameter tl. Default: 0.0
        init_ar (float, optional): Initial value for the right slope parameter ar. Default: 1.0
        init_al (float, optional): Initial value for the left slope parameter al. Default: 0.1
        fix_init_epochs (int, optional): Number of epochs to keep parameters fixed at initialization. Default: 0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SReLU(init_tr=1.0, init_tl=0.0, init_ar=1.0, init_al=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, init_tr: float = 1.0, init_tl: float = 0.0, 
                 init_ar: float = 1.0, init_al: float = 0.1,
                 fix_init_epochs: int = 0):
        super(SReLU, self).__init__()
        self.tr = nn.Parameter(Tensor([init_tr]))
        self.tl = nn.Parameter(Tensor([init_tl]))
        self.ar = nn.Parameter(Tensor([init_ar]))
        self.al = nn.Parameter(Tensor([init_al]))
        
        # For tracking epochs if parameters should be fixed initially
        self.fix_init_epochs = fix_init_epochs
        self.current_epoch = 0
        self.init_tr = init_tr
        self.init_tl = init_tl
        self.init_ar = init_ar
        self.init_al = init_al

    def forward(self, x) -> Tensor:
        # Use fixed parameters during initial epochs if specified
        if self.current_epoch < self.fix_init_epochs and self.training:
            tr = self.init_tr
            tl = self.init_tl
            ar = self.init_ar
            al = self.init_al
        else:
            tr = self.tr
            tl = self.tl
            ar = self.ar
            al = self.al
        
        # Create masks for the three regions
        right_mask = x >= tr
        left_mask = x <= tl
        middle_mask = ~(right_mask | left_mask)
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply the three piecewise linear functions
        if right_mask.any():
            result[right_mask] = tr + ar * (x[right_mask] - tr)
        
        if middle_mask.any():
            result[middle_mask] = x[middle_mask]
        
        if left_mask.any():
            result[left_mask] = tl + al * (x[left_mask] - tl)
        
        return result
    
    def train(self, mode=True):
        super(SReLU, self).train(mode)
        if mode:
            # Increment epoch counter when switching to training mode
            self.current_epoch += 1
        return self


@register_activation
class NActivation(nn.Module):
    r"""
    Applies the N-Activation function:

    :math:`\text{N-Activation}(z_i) = \begin{cases} 
    z_i - 2t_{i,min}, & z_i < t_{i,min} \\
    -z_i, & t_{i,min} \leq z_i \leq t_{i,max} \\
    z_i - 2t_{i,max}, & z_i > t_{i,max}
    \end{cases}`

    where :math:`t_{i,min} = \min(a_i, b_i)` and :math:`t_{i,max} = \max(a_i, b_i)`,
    and :math:`a_i` and :math:`b_i` are trainable parameters.

    Args:
        init_a (float, optional): Initial value for parameter a. Default: -0.5
        init_b (float, optional): Initial value for parameter b. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = NActivation(init_a=-0.5, init_b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, init_a: float = -0.5, init_b: float = 0.5):
        super(NActivation, self).__init__()
        self.a = nn.Parameter(Tensor([init_a]))
        self.b = nn.Parameter(Tensor([init_b]))

    def forward(self, x) -> Tensor:
        # Calculate t_min and t_max
        t_min = torch.min(self.a, self.b)
        t_max = torch.max(self.a, self.b)
        
        # Create masks for the three regions
        left_mask = x < t_min
        middle_mask = (x >= t_min) & (x <= t_max)
        right_mask = x > t_max
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply the three piecewise linear functions
        if left_mask.any():
            result[left_mask] = x[left_mask] - 2 * t_min
        
        if middle_mask.any():
            result[middle_mask] = -x[middle_mask]
        
        if right_mask.any():
            result[right_mask] = x[right_mask] - 2 * t_max
        
        return result


@register_activation
class ALiSA(nn.Module):
    r"""
    Applies the Adaptive Linearized Sigmoidal Activation (ALiSA) function:

    :math:`\text{ALiSA}(z_i) = \begin{cases} 
    a^r_i z_i - a^r_i + 1, & z_i \geq 1 \\
    z_i, & 1 > z_i > 0 \\
    a^l_i z_i, & z_i \leq 0
    \end{cases}`

    where :math:`a^r_i` and :math:`a^l_i` are trainable parameters.

    Args:
        init_ar (float, optional): Initial value for the right slope parameter ar. Default: 1.0
        init_al (float, optional): Initial value for the left slope parameter al. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ALiSA(init_ar=1.0, init_al=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, init_ar: float = 1.0, init_al: float = 0.1):
        super(ALiSA, self).__init__()
        self.ar = nn.Parameter(Tensor([init_ar]))
        self.al = nn.Parameter(Tensor([init_al]))

    def forward(self, x) -> Tensor:
        # Create masks for the three regions
        right_mask = x >= 1
        left_mask = x <= 0
        middle_mask = ~(right_mask | left_mask)
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply the three piecewise linear functions
        if right_mask.any():
            result[right_mask] = self.ar * x[right_mask] - self.ar + 1
        
        if middle_mask.any():
            result[middle_mask] = x[middle_mask]
        
        if left_mask.any():
            result[left_mask] = self.al * x[left_mask]
        
        return result


@register_activation
class LiSA(nn.Module):
    r"""
    Applies the Linearized Sigmoidal Activation (LiSA) function:

    :math:`\text{LiSA}(z_i) = \begin{cases} 
    a^r z_i - a^r + 1, & z_i \geq 1 \\
    z_i, & 1 > z_i > 0 \\
    a^l z_i, & z_i \leq 0
    \end{cases}`

    where :math:`a^r` and :math:`a^l` are fixed parameters.

    Args:
        ar (float, optional): Fixed value for the right slope parameter ar. Default: 1.0
        al (float, optional): Fixed value for the left slope parameter al. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LiSA(ar=1.0, al=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, ar: float = 1.0, al: float = 0.1):
        super(LiSA, self).__init__()
        self.ar = ar
        self.al = al

    def forward(self, x) -> Tensor:
        # Create masks for the three regions
        right_mask = x >= 1
        left_mask = x <= 0
        middle_mask = ~(right_mask | left_mask)
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply the three piecewise linear functions
        if right_mask.any():
            result[right_mask] = self.ar * x[right_mask] - self.ar + 1
        
        if middle_mask.any():
            result[middle_mask] = x[middle_mask]
        
        if left_mask.any():
            result[left_mask] = self.al * x[left_mask]
        
        return result