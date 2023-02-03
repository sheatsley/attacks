"""
This module initializes the adversarial machine learning repo.
Author: Ryan Sheatsley and Blaine Hoak
Tue Nov 29 2022
"""

# import modules
import aml.attacks  # pytorch-based aml attacks
import aml.loss  # pytorch-based loss functions popular in aml
import aml.optimizer  # pytorch-based optimizers popular in aml
import aml.surface  # gradient computation for aml attacks in pytorch
import aml.traveler  # perturbation computations for aml attacks in pytorch
import subprocess  # Subprocess management

# expose known attacks
from aml.attacks import apgdce  # APGD-CE (https://arxiv.org/pdf/2003.01690.pdf)
from aml.attacks import apgddlr  # APGD-DLR (https://arxiv.org/pdf/2003.01690.pdf)
from aml.attacks import bim  # BIM (https://arxiv.org/pdf/1611.01236.pdf)
from aml.attacks import cwl2  # CW-L2 (https://arxiv.org/pdf/1608.04644.pdf)
from aml.attacks import df  # DF (https://arxiv.org/pdf/1511.04599.pdf)
from aml.attacks import fab  # FAB (https://arxiv.org/pdf/1907.02044.pdf)
from aml.attacks import jsma  # JSMA (https://arxiv.org/pdf/1511.07528.pdf)
from aml.attacks import pgd  # PGD (https://arxiv.org/pdf/1706.06083.pdf)

# compute version
__version__ = subprocess.check_output(
    ("git", "rev-parse", "--short", "HEAD"), text=True
).strip()
