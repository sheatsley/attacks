"""
This module initializes the adversarial machine learning repo.
"""
import pathlib
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
try:
    cmd = ("git", "-C", *__path__, "rev-parse", "--short", "HEAD")
    __version__ = subprocess.check_output(
        cmd, stderr=subprocess.DEVNULL, text=True
    ).strip()
except subprocess.CalledProcessError:
    with open(pathlib.Path(__file__).parent / "VERSION", "r") as f:
        __version__ = f.read().strip()
