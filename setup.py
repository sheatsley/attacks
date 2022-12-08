"""
Build script for the Adversarial Machine Learning repo
Author: Ryan Sheatsley
Date: Mon Nov 21 2022
"""
import setuptools  # Easily download, build, install, upgrade, and uninstall Python packages

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    author="Ryan Sheatsley",
    author_email="ryan@sheatsley.me",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    description="A PyTorch-based adversarial machine learning library",
    install_requires=[
        "scikit-learn",
        "torch",
    ],
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="machine-learning pytorch adversarial-machine-learning",
    name="aml",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    url="https://github.com/sheatsley/adversarial_machine_learning",
    version="0.9.1",
)
