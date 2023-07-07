"""
This module initializes the adversarial machine learning repo.
"""
import subprocess

from aml.attacks import (
    Adversary,
    Attack,
    apgdce,
    apgddlr,
    bim,
    cwl2,
    df,
    fab,
    jsma,
    pgd,
)

__all__ = (
    "Adversary",
    "Attack",
    "apgdce",
    "apgddlr",
    "bim",
    "cwl2",
    "df",
    "fab",
    "jsma",
    "pgd",
)
__version__ = subprocess.check_output(
    ("git", "-C", *__path__, "rev-parse", "--short", "HEAD"), text=True
).strip()
