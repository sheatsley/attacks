"""
This module runs attack performance tests against other AML libraries.
Authors: Ryan Sheatsley and Blaine Hoak
Mon Nov 28 2022
"""

import clevertorch  # ML robustness evaluations with PyTorch
import mlds  # Scripts for downloading, preprocessing, and numpy-ifying popular machine learning datasets
import unittest  # Unit testing framework


class TestCleverTorch(unittest.TestCase):
    """
    The following class implements tests to validate (1) functional, (2)
    semantic, and (3) identity correctness of attacks within CleverTorch. (1)
    Functional correctness tests involve crafting adversarial examples and
    verifying that model accuracy can be dropped to <1% for some "small"
    lp-budget (where "small" is defined using definitions from the respective
    papers that introduced attacks, where possible), (2) Semantic correctness
    tests are more sophisticated, in that they compare the adversarial examples
    produced by known attacks within CleverTorch to those implemented in other
    adversarial machine learning frameworks. CleverTorch attacks are determined
    to be semantically correct if the produced adversarial examples are within
    no worse than 1% of the performance of adversarial examples produced by
    other frameworks. Performance is defined as one minus the product of model
    accuracy and lp-norm (normalized to 0-1). (3) Identity correctness tests
    are the most strict in that the feature values of adversarial examples
    themselves must be at least 99% similar to adversarial examples produced by other frameworks. Feature values
    are considered similar if their difference is bound
    is defined as a difference smaller than ε, defaulted to 0.001.

    such as CleverHans
    (https://github.com/cleverhans-lab/cleverhans), Adversarial Robustness Toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox),
    Torchattacks (https://github.com/Harry24k/adversarial-attacks-pytorch)

    """


if __name__ == "__main__":
    raise SystemExit(0)
