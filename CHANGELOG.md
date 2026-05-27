# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-21

### Added
- 416 activation functions across classical and adaptive families
- Activation explorer site (Next.js static site deployed to GitHub Pages)
- Universal parametrized test suite covering all activations (8 property tests)
- `@register_activation` decorator and `get_all_activations()` public API
- Pre-commit hooks for code quality (black, flake8, isort, doc8, mypy)
- CI lint workflow via GitHub Actions

### Fixed
- 50+ formula bugs found by literature review
- 10 numerical stability issues (exp overflow, division by zero, NaN propagation)
- 12 gradient and autograd issues (broken parameter gradients, inplace bugs, stochastic in eval mode)
- Device and dtype consistency across 20 activations
- Scalar input handling in BaseActivation
- Parameter initialization semantics in 8 activations

### Changed
- Moved test structure to individual test files with template-based formula validation
- Refactored BaseActivation to support scalar inputs natively

## [0.4.0] - 2026-04-15

### Added
- Initial implementation of classical activation families (sigmoid, ReLU, tanh variants)
- Initial implementation of adaptive activation families (learnable parameter activations)
- Basic test infrastructure
- Documentation structure with Sphinx

[1.0.0]: https://github.com/hdmquan/torch_activation/releases/tag/v1.0.0
[0.4.0]: https://github.com/hdmquan/torch_activation/releases/tag/v0.4.0
