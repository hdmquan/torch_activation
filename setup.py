from torch_activation import __version__
from setuptools import setup, find_packages

setup(
    name="torch_activation",
    version=__version__,
    author="Alan Huynh",
    author_email="hdmquan@gmail.com",
    description="A collection of new activation functions for PyTorch",
    url="https://github.com/hdmquan/torch_activation",
    packages=find_packages(),
    install_requires=[
        "torch >= 1.0.0",
    ],
)
