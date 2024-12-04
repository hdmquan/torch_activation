## Available Functions

---

| **ReLU Variations**   |                                         |
| --------------------- | --------------------------------------- |
| ShiLU [[1]](#1)       | $\alpha \cdot \text{ReLU}(x) + \beta$   |
| ReLUN [[1]](#1)       | $\min(\text{ReLU}(x), n)$               |
| CReLU [[2]](#2)       | $\text{ReLU}(x) \oplus \text{ReLU}(-x)$ |
| SquaredReLU [[5]](#5) | $\text{ReLU}(x)^2$                      |
| StarReLU [[8]](#8)    | $s \cdot \text{ReLU}(x)^2 + b$          |

| **GLU Variations** |                                       |
| ------------------ | ------------------------------------- |
| ReGLU [[6]](#6)    | $\text{ReLU} (xW + b) \odot (xV + c)$ |
| GeGLU [[6]](#6)    | $\text{GeLU} (xW + b) \odot (xV + c)$ |
| SwiGLU [[6]](#6)   | $\sigma (xW + b) \odot (xV + c)$      |
| SeGLU [[11]](#11)  | $\text{SELU} (xW + b) \odot (xV + c)$ |

| **Composite Functions** |                                                                            |
| ----------------------- | -------------------------------------------------------------------------- |
| DELU [[1]](#1)          | $\text{if }  x \leqslant 0 \text{, SiLU}(x); \text{ else, } x(n-1)$        |
| DReLUs [[10]](#10)      | $\text{if }  x \leqslant 0 \text{, } \alpha (e ^ x -1); \text{ else, }  x$ |

| **Trigonometry Based** |                                                      |
| ---------------------- | ---------------------------------------------------- |
| GCU [[3]](#3)          | $x \cdot \cos(x)$                                    |
| CosLU [[1]](#1)        | $(x + \alpha \cdot \cos(\beta x)) \cdot \sigma(x)$   |
| SinLU [[9]](#9)        | $(x + \alpha \cdot \sin (\beta x)) \cdot \sigma (x)$ |

| **Others**               |                                        |
| ------------------------ | -------------------------------------- |
| ScaledSoftSign [[1]](#1) | $\frac{\alpha \cdot x}{\beta + \|x\|}$ |
| CoLU [[4]](#4)           | $\frac{x}{1-x \cdot e^{-(x + e^x)}}$   |

| **Linear Combination** |                                                     |
| ---------------------- | --------------------------------------------------- |
| LinComb [[7]](#7)      | $\sum_{i=1}^{n} w_i \cdot F_i(x)$                   |
| NormLinComb [[7]](#7)  | $\frac{\sum_{i=1}^{n} w_i \cdot F_i(x)}{\|\|W\|\|}$ |

---

## References

<a id="1">[1]</a>
Pishchik, E. (2023). Trainable Activations for Image Classification. Preprints.org, 2023010463. DOI: 10.20944/preprints202301.0463.v1.

<a id="2">[2]</a>
Shang, W., Sohn, K., Almeida, D., Lee, H. (2016). Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units. arXiv:1603.05201v2 (cs).

<a id="3">[3]</a>
Noel, M. M., Arunkumar, L., Trivedi, A., Dutta, P. (2023). Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks. arXiv:2108.12943v3 (cs).

<a id="4">[4]</a>
Vagerwal, A. (2021). Deeper Learning with CoLU Activation. arXiv:2112.12078v1 (cs).

<a id="5">[5]</a>
So, D. R., Mańke, W., Liu, H., Dai, Z., Shazeer, N., Le, Q. V. (2022). Primer: Searching for Efficient Transformers for Language Modeling. arXiv:2109.08668v2 (cs)

<a id="6">[6]</a>
Noam, S. (2020). GLU Variants Improve Transformer. arXiv:2002.05202v1 (cs)

<a id="7">[7]</a>
Pishchik, E. (2023). Trainable Activations for Image Classification. Preprints.org, 2023010463. DOI: 10.20944/preprints202301.0463.v1

<a id="8">[8]</a>
Weihao, Y., et al (2022). MetaFormer Baselines for Vision. arXiv:2210.13452v2 (cs)

<a id="9">[9]</a>
Paul, A., et al (2022). SinLU: Sinu-Sigmoidal Linear Unit. DOI: 10.3390/math10030337

<a id="10">[10]</a>
Godin, F., et al (2017). Dual Rectified Linear Units (DReLUs): A Replacement for Tanh Activation Functions in Quasi-Recurrent Neural Networks. arXiv: 1707.08214v2 (cs.CL)

<a id="11">[11]</a>
Pouya, A., & Pegah, A. (2022). ActTensor (Version 1.0.0) [Computer software]. https://github.com/pouyaardehkhani/ActTensor

[Back to top](#Installation)
