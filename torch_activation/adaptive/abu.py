import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

from torch_activation import register_activation

@register_activation
class ABU(nn.Module):
    r"""
    Applies the Adaptive Blending Unit (ABU) function:

    :math:`\text{ABU}(z_l) = \sum_{j=0}^{n} a_{j,l} \cdot g_j(z_l) + b`

    where :math:`g_j(z_l)` is an activation function from a pool of n activation functions,
    :math:`a_{j,l}` is a trainable weighting parameter for each layer l and activation function g_j,
    and :math:`b` is an optional trainable bias term.

    Args:
        activation_pool (list, optional): List of activation functions to blend. 
            Default: [nn.Tanh(), nn.ELU(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        constrain_weights (str, optional): Method to constrain weights. Options: 'none', 'sum_to_one', 
            'abs_sum_to_one', 'clip_and_normalize', 'softmax'. Default: 'none'
        init_weights (list, optional): Initial weights for each activation. If None, initialized to 1/n. Default: None
        bias (bool, optional): If True, adds a learnable bias term. Default: False
        init_bias (float, optional): Initial value for the bias term. Default: 0.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ABU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, activation_pool=None, constrain_weights='none', init_weights=None, bias=False, init_bias=0.0):
        super(ABU, self).__init__()
        
        # Default activation pool if none provided
        if activation_pool is None:
            activation_pool = [nn.Tanh(), nn.ELU(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        
        self.activation_pool = nn.ModuleList(activation_pool)
        self.n_activations = len(activation_pool)
        self.constrain_weights = constrain_weights
        
        # Initialize weights
        if init_weights is None:
            init_weights = [1.0 / self.n_activations] * self.n_activations
        else:
            assert len(init_weights) == self.n_activations, "Number of initial weights must match number of activations"
        
        self.weights = nn.Parameter(torch.tensor(init_weights))
        
        # Initialize bias if needed
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.tensor(init_bias))

    def forward(self, x) -> Tensor:
        # Apply constraint to weights if needed
        if self.constrain_weights == 'sum_to_one':
            weights = self.weights / (torch.sum(self.weights) + 1e-6)
        elif self.constrain_weights == 'abs_sum_to_one':
            weights = self.weights / (torch.sum(torch.abs(self.weights)) + 1e-6)
        elif self.constrain_weights == 'clip_and_normalize':
            weights = torch.clamp(self.weights, min=0.0)
            weights = weights / (torch.sum(weights) + 1e-6)
        elif self.constrain_weights == 'softmax':
            weights = F.softmax(self.weights, dim=0)
        else:  # 'none'
            weights = self.weights
        
        # Apply each activation and blend
        result = 0
        for i, activation in enumerate(self.activation_pool):
            result = result + weights[i] * activation(x)
        
        # Add bias if enabled
        if self.bias:
            result = result + self.bias_param
        
        return result


@register_activation
class MoGU(nn.Module):
    r"""
    Applies the Mixture of Gaussian Unit (MoGU) function:

    :math:`\text{MoGU}(z_i) = \sum_{j=0}^{n} a_{i,j} \frac{1}{\sqrt{2\pi\sigma_{i,j}^2}} \exp\left(-\frac{(z_i-\mu_{i,j})^2}{2\sigma_{i,j}^2}\right)`

    where :math:`a_{i,j}`, :math:`\sigma_{i,j}`, and :math:`\mu_{i,j}` are trainable parameters.

    Args:
        n_gaussians (int, optional): Number of Gaussian components in the mixture. Default: 3
        init_a (float, optional): Initial value for the scale parameters a. Default: 1.0
        init_sigma (float, optional): Initial value for the standard deviation parameters sigma. Default: 1.0
        init_mu_spread (float, optional): Spread for initializing the mean parameters mu. Default: 2.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MoGU(n_gaussians=3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_gaussians=3, init_a=1.0, init_sigma=1.0, init_mu_spread=2.0):
        super(MoGU, self).__init__()
        self.n_gaussians = n_gaussians
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.full((n_gaussians,), init_a))
        
        # Use softplus to ensure sigma is positive
        sigma_raw = torch.full((n_gaussians,), math.log(math.exp(init_sigma) - 1))
        self.sigma_raw = nn.Parameter(sigma_raw)
        
        # Initialize means to be spread out
        mu_init = torch.linspace(-init_mu_spread, init_mu_spread, n_gaussians)
        self.mu = nn.Parameter(mu_init)

    def forward(self, x) -> Tensor:
        # Ensure sigma is positive using softplus
        sigma = F.softplus(self.sigma_raw)
        
        # Calculate the Gaussian mixture
        result = torch.zeros_like(x)
        for j in range(self.n_gaussians):
            # Calculate Gaussian component
            gaussian = torch.exp(-0.5 * ((x - self.mu[j]) / sigma[j])**2)
            gaussian = gaussian / (math.sqrt(2 * math.pi) * sigma[j])
            
            # Add weighted component to result
            result = result + self.a[j] * gaussian
        
        return result


@register_activation
class FSA(nn.Module):
    r"""
    Applies the Fourier Series Activation (FSA) function:

    :math:`\text{FSA}(z_i) = a_i + \sum_{j=1}^{r} (b_{i,j} \cos(jd_i z_i) + c_{i,j} \sin(jd_i z_i))`

    where :math:`a_i`, :math:`b_{i,j}`, :math:`c_{i,j}`, :math:`d_i` are trainable parameters,
    and :math:`r` is a fixed hyperparameter denoting the rank of the Fourier series.

    Args:
        rank (int, optional): Rank of the Fourier series (r). Default: 5
        init_a (float, optional): Initial value for the bias parameter a. Default: 0.0
        init_b (float, optional): Initial value for the cosine coefficients b. Default: 0.1
        init_c (float, optional): Initial value for the sine coefficients c. Default: 0.1
        init_d (float, optional): Initial value for the frequency parameter d. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FSA(rank=5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, rank=5, init_a=0.0, init_b=0.1, init_c=0.1, init_d=1.0):
        super(FSA, self).__init__()
        self.rank = rank
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.tensor([init_a]))
        self.b = nn.Parameter(torch.full((rank,), init_b))
        self.c = nn.Parameter(torch.full((rank,), init_c))
        self.d = nn.Parameter(torch.tensor([init_d]))

    def forward(self, x) -> Tensor:
        result = self.a.expand_as(x)
        
        for j in range(1, self.rank + 1):
            # Calculate j*d*x for each term
            angle = j * self.d * x
            
            # Add cosine and sine terms
            result = result + self.b[j-1] * torch.cos(angle) + self.c[j-1] * torch.sin(angle)
        
        return result


@register_activation
class TCA(nn.Module):
    r"""
    Applies the Trainable Compound Activation (TCA) function:

    :math:`\text{TCA}(z_i) = \frac{1}{k} \sum_{j=1}^{k} f_j(\exp(a_{i,j}) z_i + b_{i,j})`

    where :math:`k` is the number of mixed functions, and :math:`a_{i,j}` and :math:`b_{i,j}` 
    are scaling and translation trainable parameters.

    Args:
        activation_pool (list, optional): List of activation functions to mix. 
            Default: [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        init_a (float, optional): Initial value for the scaling parameters a. Default: 0.0
        init_b (float, optional): Initial value for the translation parameters b. Default: 0.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TCA()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, activation_pool=None, init_a=0.0, init_b=0.0):
        super(TCA, self).__init__()
        
        # Default activation pool if none provided
        if activation_pool is None:
            activation_pool = [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        
        self.activation_pool = nn.ModuleList(activation_pool)
        self.k = len(activation_pool)
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.full((self.k,), init_a))
        self.b = nn.Parameter(torch.full((self.k,), init_b))

    def forward(self, x) -> Tensor:
        result = 0
        for j, activation in enumerate(self.activation_pool):
            # Apply horizontal scaling and translation
            scaled_input = torch.exp(self.a[j]) * x + self.b[j]
            result = result + activation(scaled_input)
        
        # Average the results
        result = result / self.k
        
        return result


@register_activation
class TCAv2(nn.Module):
    r"""
    Applies the Trainable Compound Activation Variant 2 (TCAv2) function:

    :math:`\text{TCAv2}(z_i) = \frac{\sum_{j=1}^{k} \exp(a_{i,j}) f_j(\exp(b_{i,j}) z_i + c_{i,j})}{\sum_{j=1}^{k} \exp(a_{i,j})}`

    where :math:`k` is the number of mixed functions, and :math:`a_{i,j}`, :math:`b_{i,j}`, and :math:`c_{i,j}`
    are scaling and translation trainable parameters.

    Args:
        activation_pool (list, optional): List of activation functions to mix. 
            Default: [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        init_a (float, optional): Initial value for the vertical scaling parameters a. Default: 0.0
        init_b (float, optional): Initial value for the horizontal scaling parameters b. Default: 0.0
        init_c (float, optional): Initial value for the translation parameters c. Default: 0.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TCAv2()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, activation_pool=None, init_a=0.0, init_b=0.0, init_c=0.0):
        super(TCAv2, self).__init__()
        
        # Default activation pool if none provided
        if activation_pool is None:
            activation_pool = [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        
        self.activation_pool = nn.ModuleList(activation_pool)
        self.k = len(activation_pool)
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.full((self.k,), init_a))
        self.b = nn.Parameter(torch.full((self.k,), init_b))
        self.c = nn.Parameter(torch.full((self.k,), init_c))

    def forward(self, x) -> Tensor:
        numerator = 0
        denominator = torch.sum(torch.exp(self.a))
        
        for j, activation in enumerate(self.activation_pool):
            # Apply horizontal scaling and translation
            scaled_input = torch.exp(self.b[j]) * x + self.c[j]
            # Apply vertical scaling
            numerator = numerator + torch.exp(self.a[j]) * activation(scaled_input)
        
        result = numerator / denominator
        
        return result


@register_activation
class APAF(nn.Module):
    r"""
    Applies the Average of a Pool of Activation Functions (APAF):

    :math:`\text{APAF}(z_i) = \frac{\sum_{j=0}^{n} a_{j,i} h_j(z_i)}{\sum_{j=0}^{n} a_{j,i}}`

    where :math:`h_j` are activation functions from a pool and :math:`a_{j,i}` are trainable parameters.

    Args:
        activation_pool (list, optional): List of activation functions to average. 
            Default: [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Identity()]
        init_weights (float, optional): Initial value for the weights. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = APAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, activation_pool=None, init_weights=1.0):
        super(APAF, self).__init__()
        
        # Default activation pool if none provided
        if activation_pool is None:
            activation_pool = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Identity()]
        
        self.activation_pool = nn.ModuleList(activation_pool)
        self.n = len(activation_pool)
        
        # Initialize trainable parameters
        self.weights = nn.Parameter(torch.full((self.n,), init_weights))

    def forward(self, x) -> Tensor:
        numerator = 0
        denominator = torch.sum(self.weights)
        
        for j, activation in enumerate(self.activation_pool):
            numerator = numerator + self.weights[j] * activation(x)
        
        result = numerator / denominator
        
        return result


@register_activation
class GABU(nn.Module):
    r"""
    Applies the Gating Adaptive Blending Unit (GABU) function:

    :math:`\text{GABU}(z_i) = \sum_{j=0}^{n} \sigma(a_{j,i}) g_j(z_i)`

    where :math:`g_j` are activation functions from a pool, :math:`\sigma` is the logistic sigmoid function,
    and :math:`a_{j,i}` are trainable parameters controlling the weight of each activation function.

    Args:
        activation_pool (list, optional): List of activation functions to blend. 
            Default: [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        init_gates (float, optional): Initial value for the gating parameters. Default: 0.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = GABU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, activation_pool=None, init_gates=0.0):
        super(GABU, self).__init__()
        
        # Default activation pool if none provided
        if activation_pool is None:
            activation_pool = [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        
        self.activation_pool = nn.ModuleList(activation_pool)
        self.n = len(activation_pool)
        
        # Initialize trainable parameters
        self.gates = nn.Parameter(torch.full((self.n,), init_gates))

    def forward(self, x) -> Tensor:
        result = 0
        
        for j, activation in enumerate(self.activation_pool):
            # Apply sigmoid gating
            gate = torch.sigmoid(self.gates[j])
            result = result + gate * activation(x)
        
        return result


@register_activation
class DKNN(nn.Module):
    r"""
    Applies the Deep Kronecker Neural Network (DKNN) activation function:

    :math:`\text{DKNN}(z_l) = \sum_{j=0}^{n} a_{l,j} g_j(b_{l,j} z_l)`

    where :math:`g_j` are fixed activation functions, and :math:`a_{l,j}` and :math:`b_{l,j}` 
    are trainable parameters.

    Args:
        activation_pool (list, optional): List of activation functions to use. 
            Default: [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        init_a (float, optional): Initial value for the vertical scaling parameters a. Default: 1.0
        init_b (float, optional): Initial value for the horizontal scaling parameters b. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = DKNN()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, activation_pool=None, init_a=1.0, init_b=1.0):
        super(DKNN, self).__init__()
        
        # Default activation pool if none provided
        if activation_pool is None:
            activation_pool = [nn.Tanh(), nn.ReLU(), nn.SiLU(), nn.Identity()]
        
        self.activation_pool = nn.ModuleList(activation_pool)
        self.n = len(activation_pool)
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.full((self.n,), init_a))
        self.b = nn.Parameter(torch.full((self.n,), init_b))

    def forward(self, x) -> Tensor:
        result = 0
        
        for j, activation in enumerate(self.activation_pool):
            # Apply horizontal scaling and vertical scaling
            result = result + self.a[j] * activation(self.b[j] * x)
        
        return result


@register_activation
class RowdyActivation(nn.Module):
    r"""
    Applies the Rowdy Activation function, a special case of DKNN:

    :math:`\text{Rowdy}(z_l) = g_0(z_l) + \sum_{j=1}^{n} a_j \cdot c \cdot \sin(jcz_l)`

    where :math:`g_0` is a base activation function, :math:`c` is a fixed scaling factor,
    and :math:`a_j` are trainable parameters.

    Args:
        base_activation (nn.Module, optional): Base activation function g_0. Default: nn.ReLU()
        n_terms (int, optional): Number of sine terms to use. Default: 5
        scaling_factor (float, optional): Fixed scaling factor c. Default: 1.0
        init_a (float, optional): Initial value for the scaling parameters a. Default: 0.1
        use_cos (bool, optional): If True, uses cosine instead of sine. Default: False

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = RowdyActivation(n_terms=3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, base_activation=None, n_terms=5, scaling_factor=1.0, init_a=0.1, use_cos=False):
        super(RowdyActivation, self).__init__()
        
        if base_activation is None:
            base_activation = nn.ReLU()
        
        self.base_activation = base_activation
        self.n_terms = n_terms
        self.c = scaling_factor
        self.use_cos = use_cos
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.full((n_terms,), init_a))

    def forward(self, x) -> Tensor:
        result = self.base_activation(x)
        
        for j in range(1, self.n_terms + 1):
            angle = j * self.c * x
            if self.use_cos:
                result = result + self.a[j-1] * self.c * torch.cos(angle)
            else:
                result = result + self.a[j-1] * self.c * torch.sin(angle)
        
        return result


@register_activation
class SLAF(nn.Module):
    r"""
    Applies the Self-Learnable Activation Function (SLAF):

    :math:`\text{SLAF}(z_i) = \sum_{j=0}^{k-1} a_{i,j} z_i^j`

    where :math:`a_{i,j}` are learnable parameters for each neuron and :math:`k` is a hyperparameter
    defining the number of elements in the polynomial expression.

    Args:
        k (int, optional): Number of terms in the polynomial. Default: 6
        init_a (float, optional): Initial value for the coefficients. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SLAF(k=4)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, k=6, init_a=0.1):
        super(SLAF, self).__init__()
        self.k = k
        
        # Initialize trainable parameters
        self.a = nn.Parameter(torch.full((k,), init_a))

    def forward(self, x) -> Tensor:
        result = self.a[0] * torch.ones_like(x)  # j=0 term
        
        # Compute powers of x and multiply by coefficients
        x_power = x  # Start with x^1
        for j in range(1, self.k):
            result = result + self.a[j] * x_power
            x_power = x_power * x  # Compute next power
        
        return result


@register_activation
class ChPAF(nn.Module):
    r"""
    Applies the Chebyshev Polynomial-based Activation Function (ChPAF):

    :math:`\text{ChPAF}(z) = \sum_{j=0}^{k} a_j C_j(z)`

    where :math:`a_j` are learnable parameters, :math:`k` is a fixed hyperparameter denoting the
    maximum order of used Chebyshev polynomials, and :math:`C_j(z)` is a Chebyshev polynomial of order j.

    Args:
        k (int, optional): Maximum order of Chebyshev polynomials. Default: 3
        init_a (float, optional): Initial value for the coefficients. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ChPAF(k=3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, k=3, init_a=0.1):
        super(ChPAF, self).__init__()
        self.k = k
        
        # Initialize trainable parameters
        self.coefficients = nn.Parameter(torch.full((k+1,), init_a))
        
    def forward(self, x) -> Tensor:
        result = self.coefficients[0]  # C_0(x) = 1
        
        if self.k >= 1:
            # C_1(x) = x
            c_prev = torch.ones_like(x)
            c_curr = x
            result = result + self.coefficients[1] * c_curr
            
            # Higher order Chebyshev polynomials using recurrence relation
            # C_{j+1}(x) = 2x*C_j(x) - C_{j-1}(x)
            for j in range(1, self.k):
                c_next = 2 * x * c_curr - c_prev
                result = result + self.coefficients[j+1] * c_next
                c_prev, c_curr = c_curr, c_next
        
        return result


@register_activation
class LPAF(nn.Module):
    r"""
    Applies the Legendre Polynomial-based Activation Function (LPAF):

    :math:`\text{LPAF}(z) = \sum_{j=0}^{k} a_j G_j(z)`

    where :math:`a_j` are learnable parameters, :math:`k` is a fixed hyperparameter denoting the
    maximum order of used Legendre polynomials, and :math:`G_j(z)` is a Legendre polynomial of order j.

    Args:
        k (int, optional): Maximum order of Legendre polynomials. Default: 3
        init_a (float, optional): Initial value for the coefficients. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LPAF(k=3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, k=3, init_a=0.1):
        super(LPAF, self).__init__()
        self.k = k
        
        # Initialize trainable parameters
        self.coefficients = nn.Parameter(torch.full((k+1,), init_a))
        
    def forward(self, x) -> Tensor:
        result = self.coefficients[0]  # G_0(x) = 1
        
        if self.k >= 1:
            # G_1(x) = x
            g_prev = torch.ones_like(x)
            g_curr = x
            result = result + self.coefficients[1] * g_curr
            
            # Higher order Legendre polynomials using recurrence relation
            # G_{j+1}(x) = ((2j+1)/(j+1))x*G_j(x) - (j/(j+1))*G_{j-1}(x)
            for j in range(1, self.k):
                factor1 = (2*j + 1) / (j + 1)
                factor2 = j / (j + 1)
                g_next = factor1 * x * g_curr - factor2 * g_prev
                result = result + self.coefficients[j+1] * g_next
                g_prev, g_curr = g_curr, g_next
        
        return result


@register_activation
class HPAF(nn.Module):
    r"""
    Applies the Hermite Polynomial-based Activation Function (HPAF):

    :math:`\text{HPAF}(z) = \sum_{j=0}^{k} a_j H_j(z)`

    where :math:`a_j` are learnable parameters, :math:`k` is a fixed hyperparameter denoting the
    maximum order of used Hermite polynomials, and :math:`H_j(z)` is a Hermite polynomial of order j.

    Args:
        order (int, optional): Maximum order of Hermite polynomials. Default: 5
        init_a (float, optional): Initial value for the coefficients. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HPAF(order=5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, order=5, init_a=0.1):
        super(HPAF, self).__init__()
        self.order = order
        
        # Initialize trainable parameters
        self.coefficients = nn.Parameter(torch.full((order+1,), init_a))
        
    def forward(self, x) -> Tensor:
        result = self.coefficients[0]  # H_0(x) = 1
        
        if self.order >= 1:
            # H_1(x) = x
            h_prev = torch.ones_like(x)
            h_curr = x
            result = result + self.coefficients[1] * h_curr
            
            # Higher order Hermite polynomials using recurrence relation
            # H_{n+1}(x) = x*H_n(x) - n*H_{n-1}(x)
            for n in range(1, self.order):
                h_next = x * h_curr - n * h_prev
                result = result + self.coefficients[n+1] * h_next
                h_prev, h_curr = h_curr, h_next
        
        return result