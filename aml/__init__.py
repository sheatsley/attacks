"""
This module initializes the adversarial machine learning repo.
Author: Ryan Sheatsley and Blaine Hoak
Tue Nov 29 2022
"""
import subprocess  # Subprocess management

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
