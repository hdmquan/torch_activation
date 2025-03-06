import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from scipy.special import bernoulli

from torch_activation import register_activation

# TODO: Naming

class FAAF(nn.Module):
    r"""
    Applies the Fractional Adaptive Activation Function (FAAF):

    :math:`\text{FAAF}(z) = D^a f(z)`

    where :math:`D^a` is the a-th fractional derivative of function f.

    This is a base class for fractional activation functions.

    Args:
        a_init (float, optional): Initial value for the fractional order parameter a. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a_init: float = 0.5):
        super(FAAF, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))

    def forward(self, x) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


@register_activation
class FracReLU(nn.Module):
    r"""
    Applies the Fractional ReLU function:

    :math:`\text{FracReLU}(z_i) = \frac{z_i^{1 - a_i}}{\Gamma(2 - a_i)}`

    where :math:`\Gamma` is the Gamma function and :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracReLU(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, eps: float = 1e-6):
        super(FracReLU, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.eps = eps

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Only apply to positive values (like ReLU)
        pos_mask = x > 0
        result = torch.zeros_like(x)
        
        # Apply fractional operation only to positive values
        if pos_mask.any():
            x_pos = x[pos_mask]
            # Add eps for numerical stability when x is close to zero
            x_pos = x_pos + self.eps
            gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
            result[pos_mask] = torch.pow(x_pos, 1 - a_clamped) / gamma_term
            
        return result


@register_activation
class FracSoftplus(nn.Module):
    r"""
    Applies the Fractional Softplus function:

    :math:`\text{FracSoftplus}(z_i) = D^{a_i} \ln(1 + \exp(z_i))`

    Approximated using a finite sum:

    :math:`\text{FracSoftplus}(z_i) \approx \frac{1}{h^{a_i}} \sum_{n=0}^{N} \frac{(-1)^n \Gamma(a_i + 1) \ln(1 + \exp(z_i - nh))}{\Gamma(n + 1) \Gamma(1 - n + a_i)}`

    where :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        h (float, optional): Step size for approximation. Default: 0.1
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracSoftplus(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, h: float = 0.1, n_terms: int = 5):
        super(FracSoftplus, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.h = h
        self.n_terms = n_terms

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 2) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 1.99)
        
        # Special cases for efficiency
        if torch.abs(a_clamped - 0.0) < 1e-4:
            # When a ≈ 0, return softplus
            return F.softplus(x)
        elif torch.abs(a_clamped - 1.0) < 1e-4:
            # When a ≈ 1, return sigmoid
            return torch.sigmoid(x)
        
        # Compute the approximation using finite sum
        result = torch.zeros_like(x)
        h_pow_a = torch.pow(self.h, a_clamped)
        gamma_a_plus_1 = torch.exp(torch.lgamma(a_clamped + 1))
        
        for n in range(self.n_terms):
            # Calculate terms in the sum
            coef = ((-1) ** n) * gamma_a_plus_1
            denom = math.factorial(n) * torch.exp(torch.lgamma(1 - n + a_clamped))
            term = coef * F.softplus(x - n * self.h) / denom
            result += term
            
        return result / h_pow_a


@register_activation
class FracTanh(nn.Module):
    r"""
    Applies the Fractional Hyperbolic Tangent function:

    :math:`\text{FracTanh}(z_i) = D^{a_i} \tanh(z_i)`

    Approximated using a finite sum:

    :math:`\text{FracTanh}(z_i) \approx \frac{1}{h^{a_i}} \sum_{n=0}^{N} \frac{(-1)^n \Gamma(a_i + 1) \tanh(z_i - nh)}{\Gamma(n + 1) \Gamma(1 - n + a_i)}`

    where :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        h (float, optional): Step size for approximation. Default: 0.1
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracTanh(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, h: float = 0.1, n_terms: int = 5):
        super(FracTanh, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.h = h
        self.n_terms = n_terms

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 2) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 1.99)
        
        # Special cases for efficiency
        if torch.abs(a_clamped - 0.0) < 1e-4:
            # When a ≈ 0, return tanh
            return torch.tanh(x)
        
        # Compute the approximation using finite sum
        result = torch.zeros_like(x)
        h_pow_a = torch.pow(self.h, a_clamped)
        gamma_a_plus_1 = torch.exp(torch.lgamma(a_clamped + 1))
        
        for n in range(self.n_terms):
            # Calculate terms in the sum
            coef = ((-1) ** n) * gamma_a_plus_1
            denom = math.factorial(n) * torch.exp(torch.lgamma(1 - n + a_clamped))
            term = coef * torch.tanh(x - n * self.h) / denom
            result += term
            
        return result / h_pow_a


@register_activation
class FALU(nn.Module):
    r"""
    Applies the Fractional Adaptive Linear Unit (FALU):

    :math:`\text{FALU}(z_i) = D^{a_i} z_i \sigma(b_i z_i)`

    where :math:`\sigma` is the sigmoid function, and :math:`a_i` and :math:`b_i` are trainable parameters.

    For computational efficiency, an approximation is used:

    :math:`\text{FALU}(z_i) \approx \begin{cases} 
    g(z_i, b_i) + a_i \sigma(b_i z_i) (1 - g(z_i, b_i)), & a_i \in [0, 1] \\
    g(z_i, b_i) + a_i \sigma(b_i z_i) (1 - 2h(z_i, b_i)), & a_i \in (1, 2] 
    \end{cases}`

    where:
    :math:`g(z_i, b_i) = z_i \sigma(b_i z_i)`
    :math:`h(z_i, b_i) = g(z_i, b_i) + \sigma(z_i) (1 - g(z_i, b_i))`

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        b_init (float, optional): Initial value for the trainable parameter b. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FALU(a_init=0.3, b_init=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, b_init: float = 1.0):
        super(FALU, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.b = nn.Parameter(Tensor([b_init]))

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 2) range and b to be in (1, 10) range for stability
        a_clamped = torch.clamp(self.a, 0.0, 2.0)
        b_clamped = torch.clamp(self.b, 1.0, 10.0)
        
        # Calculate g(z, b) = z * sigmoid(b*z)
        g_z_b = x * torch.sigmoid(b_clamped * x)
        
        # Calculate h(z, b) = g(z, b) + sigmoid(z) * (1 - g(z, b))
        h_z_b = g_z_b + torch.sigmoid(x) * (1 - g_z_b)
        
        # Apply the approximation based on the value of a
        result = torch.zeros_like(x)
        
        # For a in [0, 1]
        mask_a_0_1 = (a_clamped >= 0) & (a_clamped <= 1)
        if mask_a_0_1:
            result = g_z_b + a_clamped * torch.sigmoid(b_clamped * x) * (1 - g_z_b)
        
        # For a in (1, 2]
        mask_a_1_2 = (a_clamped > 1) & (a_clamped <= 2)
        if mask_a_1_2:
            result = g_z_b + a_clamped * torch.sigmoid(b_clamped * x) * (1 - 2 * h_z_b)
            
        return result


@register_activation
class FracLReLU(nn.Module):
    r"""
    Applies the Fractional Leaky ReLU function:

    :math:`\text{FracLReLU}(z_i) = \begin{cases} 
    \frac{z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i \geq 0 \\
    \frac{0.1 \cdot z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i < 0 
    \end{cases}`

    where :math:`\Gamma` is the Gamma function and :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        negative_slope (float, optional): Controls the angle of the negative slope. Default: 0.1
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracLReLU(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, negative_slope: float = 0.1, eps: float = 1e-6):
        super(FracLReLU, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.negative_slope = negative_slope
        self.eps = eps

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Calculate gamma term
        gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
        
        # Create masks for positive and negative values
        pos_mask = x >= 0
        neg_mask = x < 0
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply to positive values
        if pos_mask.any():
            x_pos = x[pos_mask]
            # Add eps for numerical stability when x is close to zero
            x_pos = x_pos + self.eps
            result[pos_mask] = torch.pow(x_pos, 1 - a_clamped) / gamma_term
        
        # Apply to negative values
        if neg_mask.any():
            x_neg = -x[neg_mask]  # Make positive for power operation
            # Add eps for numerical stability when x is close to zero
            x_neg = x_neg + self.eps
            result[neg_mask] = -self.negative_slope * torch.pow(x_neg, 1 - a_clamped) / gamma_term
            
        return result


@register_activation
class FracPReLU(nn.Module):
    r"""
    Applies the Fractional Parametric ReLU function:

    :math:`\text{FracPReLU}(z_i) = \begin{cases} 
    \frac{z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i \geq 0 \\
    \frac{b_i \cdot z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i < 0 
    \end{cases}`

    where :math:`\Gamma` is the Gamma function, :math:`a_i` is a fixed parameter, and :math:`b_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the parameter a. Default: 0.5
        b_init (float, optional): Initial value for the trainable parameter b. Default: 0.25
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracPReLU(a_init=0.3, b_init=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, b_init: float = 0.25, eps: float = 1e-6):
        super(FracPReLU, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.b = nn.Parameter(Tensor([b_init]))
        self.eps = eps

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Calculate gamma term
        gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
        
        # Create masks for positive and negative values
        pos_mask = x >= 0
        neg_mask = x < 0
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply to positive values
        if pos_mask.any():
            x_pos = x[pos_mask]
            # Add eps for numerical stability when x is close to zero
            x_pos = x_pos + self.eps
            result[pos_mask] = torch.pow(x_pos, 1 - a_clamped) / gamma_term
        
        # Apply to negative values
        if neg_mask.any():
            x_neg = -x[neg_mask]  # Make positive for power operation
            # Add eps for numerical stability when x is close to zero
            x_neg = x_neg + self.eps
            # Use absolute value of b for stability
            b_abs = torch.abs(self.b)
            result[neg_mask] = -b_abs * torch.pow(x_neg, 1 - a_clamped) / gamma_term
            
        return result


@register_activation
class FracELU(nn.Module):
    r"""
    Applies the Fractional Exponential Linear Unit function:

    :math:`\text{FracELU}(z_i) = \begin{cases} 
    \frac{z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i \geq 0 \\
    b \sum_{k=0}^{N} \frac{1}{\Gamma(k + 1)} \frac{\Gamma(k + 1 - a_i)}{\Gamma(k + 1)} z_i^{k - a_i} - \frac{b}{\Gamma(1 - a_i)} z_i^{-a_i}, & z_i < 0 
    \end{cases}`

    where :math:`\Gamma` is the Gamma function, :math:`a_i` is a trainable parameter, and :math:`b` is a fixed parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        alpha (float, optional): The alpha parameter for ELU. Default: 1.0
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracELU(a_init=0.3, alpha=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, alpha: float = 1.0, n_terms: int = 5, eps: float = 1e-6):
        super(FracELU, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.alpha = alpha
        self.n_terms = n_terms
        self.eps = eps

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Create masks for positive and negative values
        pos_mask = x >= 0
        neg_mask = x < 0
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply to positive values
        if pos_mask.any():
            x_pos = x[pos_mask]
            # Add eps for numerical stability when x is close to zero
            x_pos = x_pos + self.eps
            gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
            result[pos_mask] = torch.pow(x_pos, 1 - a_clamped) / gamma_term
        
        # Apply to negative values
        if neg_mask.any():
            x_neg = x[neg_mask]
            
            # First term: -b/Gamma(1-a) * x^(-a)
            gamma_term_1 = torch.exp(torch.lgamma(1 - a_clamped))
            first_term = -self.alpha / gamma_term_1 * torch.pow(torch.abs(x_neg) + self.eps, -a_clamped)
            
            # Second term: b * sum(...)
            second_term = torch.zeros_like(x_neg)
            for k in range(self.n_terms):
                gamma_k_plus_1 = math.factorial(k)
                gamma_k_plus_1_minus_a = torch.exp(torch.lgamma(k + 1 - a_clamped))
                coef = 1 / gamma_k_plus_1 * gamma_k_plus_1_minus_a / gamma_k_plus_1
                term = coef * torch.pow(torch.abs(x_neg) + self.eps, k - a_clamped)
                second_term += term
            
            result[neg_mask] = self.alpha * second_term + first_term
            
        return result


@register_activation
class FracSiLU1(nn.Module):
    r"""
    Applies the Fractional SiLU Variant 1 function:

    :math:`\text{FracSiLU1}(z_i) = \begin{cases} 
    \frac{z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i \geq 0 \\
    \sum_{k=0}^{N} \frac{(-1)^k + (2^{k+1} - 1) B_{k+1} \Gamma(k + 2)}{\Gamma(k + 2 - a_i) (k + 1)!} z_i^{k+1 - a_i}, & z_i < 0 
    \end{cases}`

    where :math:`\Gamma` is the Gamma function, :math:`B_n` is the n-th Bernoulli number, and :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracSiLU1(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, n_terms: int = 5, eps: float = 1e-6):
        super(FracSiLU1, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.n_terms = n_terms
        self.eps = eps
        # Pre-compute Bernoulli numbers
        self.bernoulli_numbers = [bernoulli(i) for i in range(n_terms + 2)]

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Create masks for positive and negative values
        pos_mask = x >= 0
        neg_mask = x < 0
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply to positive values
        if pos_mask.any():
            x_pos = x[pos_mask]
            # Add eps for numerical stability when x is close to zero
            x_pos = x_pos + self.eps
            gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
            result[pos_mask] = torch.pow(x_pos, 1 - a_clamped) / gamma_term
        
        # Apply to negative values
        if neg_mask.any():
            x_neg = x[neg_mask]
            
            # Compute the sum
            sum_term = torch.zeros_like(x_neg)
            for k in range(self.n_terms):
                # Calculate (-1)^k + (2^(k+1) - 1) * B_{k+1}
                bernoulli_term = self.bernoulli_numbers[k + 1]
                coef_term = ((-1) ** k) + ((2 ** (k + 1)) - 1) * bernoulli_term
                
                # Calculate Gamma(k + 2) / (Gamma(k + 2 - a) * (k + 1)!)
                gamma_k_plus_2 = math.factorial(k + 1)
                gamma_k_plus_2_minus_a = torch.exp(torch.lgamma(k + 2 - a_clamped))
                k_plus_1_factorial = math.factorial(k + 1)
                
                coef = coef_term * gamma_k_plus_2 / (gamma_k_plus_2_minus_a * k_plus_1_factorial)
                
                # Calculate z^(k+1-a)
                power_term = torch.pow(torch.abs(x_neg) + self.eps, k + 1 - a_clamped)
                
                sum_term += coef * power_term
            
            result[neg_mask] = sum_term
            
        return result

# TODO: Forward pass failed: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
# @register_activation
class FracSiLU2(nn.Module):
    r"""
    Applies the Fractional SiLU Variant 2 function:

    :math:`\text{FracSiLU2}(z_i) = \sum_{k=0}^{N} \frac{(-1)^k + (2^{k+1} - 1) B_{k+1} \Gamma(k + 2)}{\Gamma(k + 2 - a_i) (k + 1)!} z_i^{k+1 - a_i}`

    where :math:`\Gamma` is the Gamma function, :math:`B_n` is the n-th Bernoulli number, and :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracSiLU2(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, n_terms: int = 5, eps: float = 1e-6):
        super(FracSiLU2, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.n_terms = n_terms
        self.eps = eps
        # Pre-compute Bernoulli numbers
        self.bernoulli_numbers = [bernoulli(i) for i in range(n_terms + 2)]

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Compute the sum for all values
        result = torch.zeros_like(x)
        
        for k in range(self.n_terms):
            # Calculate (-1)^k + (2^(k+1) - 1) * B_{k+1}
            bernoulli_term = self.bernoulli_numbers[k + 1]
            coef_term = ((-1) ** k) + ((2 ** (k + 1)) - 1) * bernoulli_term
            
            # Calculate Gamma(k + 2) / (Gamma(k + 2 - a) * (k + 1)!)
            gamma_k_plus_2 = math.factorial(k + 1)
            gamma_k_plus_2_minus_a = torch.exp(torch.lgamma(k + 2 - a_clamped))
            k_plus_1_factorial = math.factorial(k + 1)
            
            coef = coef_term * gamma_k_plus_2 / (gamma_k_plus_2_minus_a * k_plus_1_factorial)
            
            # Calculate z^(k+1-a)
            power_term = torch.pow(torch.abs(x) + self.eps, k + 1 - a_clamped) * torch.sign(x)
            
            result += coef * power_term
        
        return result


@register_activation
class FracGELU1(nn.Module):
    r"""
    Applies the Fractional GELU Variant 1 function:

    :math:`\text{FracGELU1}(z_i) = \begin{cases} 
    \frac{z_i^{1 - a_i}}{\Gamma(2 - a_i)}, & z_i \geq 0 \\
    \frac{0.5 z_i^{1 - a_i}}{\Gamma(2 - a_i)} - \frac{1}{\sqrt{2\pi}} \sum_{k=0}^{N} \frac{1}{k!} \left(-\frac{1}{2}\right)^k \frac{\Gamma(2k + 3)}{\Gamma(2k + 3 - a_i)} z_i^{2k+1 - a_i}, & z_i < 0 
    \end{cases}`

    where :math:`\Gamma` is the Gamma function and :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracGELU1(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, n_terms: int = 5, eps: float = 1e-6):
        super(FracGELU1, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.n_terms = n_terms
        self.eps = eps
        self.sqrt_2pi = math.sqrt(2 * math.pi)

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # Create masks for positive and negative values
        pos_mask = x >= 0
        neg_mask = x < 0
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Apply to positive values
        if pos_mask.any():
            x_pos = x[pos_mask]
            # Add eps for numerical stability when x is close to zero
            x_pos = x_pos + self.eps
            gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
            result[pos_mask] = torch.pow(x_pos, 1 - a_clamped) / gamma_term
        
        # Apply to negative values
        if neg_mask.any():
            x_neg = x[neg_mask]
            
            # First term: 0.5 * z^(1-a) / Gamma(2-a)
            gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
            first_term = 0.5 * torch.pow(torch.abs(x_neg) + self.eps, 1 - a_clamped) / gamma_term
            
            # Second term: -1/sqrt(2π) * sum(...)
            sum_term = torch.zeros_like(x_neg)
            for k in range(self.n_terms):
                # Calculate 1/k! * (-1/2)^k
                k_factorial = math.factorial(k)
                neg_half_pow_k = ((-0.5) ** k)
                
                # Calculate Gamma(2k+3) / Gamma(2k+3-a)
                gamma_2k_plus_3 = math.factorial(2*k + 2)
                gamma_2k_plus_3_minus_a = torch.exp(torch.lgamma(2*k + 3 - a_clamped))
                
                # Calculate z^(2k+1-a)
                power_term = torch.pow(torch.abs(x_neg) + self.eps, 2*k + 1 - a_clamped)
                
                term = (1 / k_factorial) * neg_half_pow_k * (gamma_2k_plus_3 / gamma_2k_plus_3_minus_a) * power_term
                sum_term += term
            
            second_term = -(1 / self.sqrt_2pi) * sum_term
            
            result[neg_mask] = first_term + second_term
            
        return result


@register_activation
class FracGELU2(nn.Module):
    r"""
    Applies the Fractional GELU Variant 2 function:

    :math:`\text{FracGELU2}(z_i) = \frac{0.5 z_i^{1 - a_i}}{\Gamma(2 - a_i)} - \frac{1}{\sqrt{2\pi}} \sum_{k=0}^{N} \frac{1}{k!} \left(-\frac{1}{2}\right)^k \frac{\Gamma(2k + 3)}{\Gamma(2k + 3 - a_i)} z_i^{2k+1 - a_i}`

    where :math:`\Gamma` is the Gamma function and :math:`a_i` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 0.5
        n_terms (int, optional): Number of terms in the approximation sum. Default: 5
        eps (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FracGELU2(a_init=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 0.5, n_terms: int = 5, eps: float = 1e-6):
        super(FracGELU2, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.n_terms = n_terms
        self.eps = eps
        self.sqrt_2pi = math.sqrt(2 * math.pi)

    def forward(self, x) -> Tensor:
        # Clamp a to be in (0, 1) range for stability
        a_clamped = torch.clamp(self.a, 0.01, 0.99)
        
        # First term: 0.5 * z^(1-a) / Gamma(2-a)
        gamma_term = torch.exp(torch.lgamma(2 - a_clamped))
        first_term = 0.5 * torch.pow(torch.abs(x) + self.eps, 1 - a_clamped) * torch.sign(x)
        first_term = first_term / gamma_term
        
        # Second term: -1/sqrt(2π) * sum(...)
        sum_term = torch.zeros_like(x)
        for k in range(self.n_terms):
            # Calculate 1/k! * (-1/2)^k
            k_factorial = math.factorial(k)
            neg_half_pow_k = ((-0.5) ** k)
            
            # Calculate Gamma(2k+3) / Gamma(2k+3-a)
            gamma_2k_plus_3 = math.factorial(2*k + 2)
            gamma_2k_plus_3_minus_a = torch.exp(torch.lgamma(2*k + 3 - a_clamped))
            
            # Calculate z^(2k+1-a)
            power_term = torch.pow(torch.abs(x) + self.eps, 2*k + 1 - a_clamped) * torch.sign(x)
            
            term = (1 / k_factorial) * neg_half_pow_k * (gamma_2k_plus_3 / gamma_2k_plus_3_minus_a) * power_term
            sum_term += term
        
        second_term = -(1 / self.sqrt_2pi) * sum_term
        
        return first_term + second_term
