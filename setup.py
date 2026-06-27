#!/usr/bin/env python
"""Setup script for aria-trl."""

from setuptools import setup, find_packages

setup(
    name="aria-trl",
    version="1.0.0",
    description="Continual learning for LLM fine-tuning via ARIA mechanisms",
    author="Darshan Poudel",
    author_email="poudeldarshan44@gmail.com",
    url="https://github.com/rsd-darshan/aria-trl",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "trl>=0.7.0",
        "datasets>=2.10.0",
        "numpy>=1.23.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
        ],
        "examples": [
            "wandb>=0.15.0",
        ],
    },
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
