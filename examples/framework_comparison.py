"""
This script compares the performance of the aml framework against other popular
frameworks and plots the model accuracy over the norm.
Author: Ryan Sheatsley
Sat Feb 4 2023
"""
import argparse
import builtins
import collections
import importlib
import time
import warnings

import aml
import dlm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import mlds
import numpy as np
import pandas
import seaborn
import torch

# dlm uses lazy modules which induce warnings that overload stdout
warnings.filterwarnings("ignore", category=UserWarning)


def apgd_art(
    alpha,
    clip_max,
    clip_min,
    epochs,
    epsilon,
    loss,
    model,
    num_restarts,
    verbose,
    x,
    **_,
):
    """
    This function crafts adversarial examples with APGD-CE or APGD-DLR in ART.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param loss: loss function to use
    :type loss: str
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of random restarts to perform
    :type num_restarts: int
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :return: APGD-CE/DLR ART adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from art.attacks.evasion import AutoProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier

    art_classifier = PyTorchClassifier(
        model=model.model,
        clip_values=(clip_min.max().item(), clip_max.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(x.size(1),),
        nb_classes=model.classes,
        device_type="gpu" if model.device == "cuda" else model.device,
    )
    return torch.from_numpy(
        AutoProjectedGradientDescent(
            estimator=art_classifier,
            norm="inf",
            eps=epsilon,
            eps_step=alpha,
            max_iter=epochs,
            targeted=False,
            nb_random_init=num_restarts,
            batch_size=x.size(0),
            loss_type="cross_entropy" if loss == "ce" else "difference_logits_ratio",
            verbose=verbose,
        ).generate(x=x.clone().cpu().numpy())
    ).to(model.device)


def apgd_torchattacks(
    epochs, epsilon, loss, model, num_restarts, rho, verbose, x, y, **_
):
    """
    This function crafts adversarial examples with APGD-CE or APGD-DLR in
    Torchattacks. Notably, the Torchattacks implementation assumes an image
    (batches, channels, width, height).

    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param loss: loss function to use
    :type loss: str
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of random restarts to perform
    :type num_restarts: int
    :param rho: rho parameter for APGD
    :type rho: float
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: APGD-CE/DLR Torchattacks adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from torchattacks import APGD

    return (
        APGD(
            model=model,
            norm="Linf",
            eps=epsilon,
            steps=epochs,
            n_restarts=num_restarts,
            seed=0,
            loss="ce" if loss == "ce" else "dlr",
            eot_iter=1,
            rho=rho,
            verbose=verbose,
        )(inputs=x.clone().unflatten(1, model.shape), labels=y)
        .flatten(1)
        .detach()
    )


def apgdce(clip_max, clip_min, frameworks, parameters, verbose, x, y, **_):
    """
    This function crafts adversarial examples with APGD-CE (Auto-PGD with CE
    loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks for
    APGD-CE include ART and Torchattacks. Notably, the Torchattacks
    implementation assumes an image (batches, channels, width, height).

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: APGD-CE adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    apgdce = aml.attacks.apgdce(**parameters)
    apgdce_params = dict(
        alpha=apgdce.alpha,
        clip_max=clip_max,
        clip_min=clip_min,
        epochs=apgdce.epochs,
        epsilon=apgdce.epsilon,
        loss="ce",
        model=apgdce.model,
        num_restarts=apgdce.num_restarts,
        rho=apgdce.traveler.optimizer.param_groups[0]["rho"],
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(ART=apgd_art, TorchAttacks=apgd_torchattacks)
    frameworks = [fw for fw in frameworks if fw in fw_func]
    "Torchattacks" in frameworks and not hasattr(
        apgdce.model, "shape"
    ) and frameworks.remove("Torchattacks")
    for fw in frameworks:
        reset_seeds()
        start = time.time()
        print(f"Producing APGD-CE adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**apgdce_params), time.time() - start, fw
    reset_seeds()
    start = time.time()
    yield (x + apgdce.craft(x, y), time.time() - start, "aml")


def apgddlr(clip_max, clip_min, frameworks, parameters, verbose, x, y, **_):
    """
    This function crafts adversarial examples with APGD-DLR (Auto-PGD with DLR
    loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks for
    APGD-DLR include ART and Torchattacks. Notably, DLR loss is undefined for
    these frameworks when there are only two classes. Moreover, the
    Torchattacks implementation assumes an image (batches, channels, width,
    height).

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: APGD-DLR adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    apgddlr = aml.attacks.apgddlr(**parameters)
    apgddlr_params = dict(
        alpha=apgddlr.alpha,
        clip_max=clip_max,
        clip_min=clip_min,
        epochs=apgddlr.epochs,
        epsilon=apgddlr.epsilon,
        loss="dlr",
        model=apgddlr.model,
        num_restarts=apgddlr.num_restarts,
        rho=apgddlr.traveler.optimizer.param_groups[0]["rho"],
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(ART=apgd_art, TorchAttacks=apgd_torchattacks)
    frameworks = [fw for fw in frameworks if fw in fw_func]
    "ART" in frameworks and apgddlr.model.classes < 3 and frameworks.remove("ART")
    "Torchattacks" in frameworks and apgddlr.model.classes < 3 and not hasattr(
        apgdce.model, "shape"
    ) and frameworks.remove("Torchattacks")
    for fw in frameworks:
        reset_seeds()
        start = time.time()
        print(f"Producing APGD-DLR adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**apgddlr_params), time.time() - start, fw
    reset_seeds()
    start = time.time()
    yield x + apgddlr.craft(x, y), time.time() - start, "aml"


def bim(clip_max, clip_min, frameworks, parameters, verbose, x, y, **_):
    """
    This function crafts adversarial examples with BIM (Basic Iterative Method)
    (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for BIM
    include AdverTorch, ART, CleverHans, Foolbox, and Torchattacks. Notably,
    CleverHans does not have an explicit implementation of BIM, so we call PGD,
    but with random initialization disabled.

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: BIM adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    bim = aml.attacks.bim(**parameters)
    bim_params = dict(
        alpha=bim.alpha,
        clip_max=clip_max,
        clip_min=clip_min,
        epochs=bim.epochs,
        epsilon=bim.epsilon,
        model=bim.model,
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(
        AdverTorch=bim_advertorch,
        ART=bim_art,
        CleverHans=bim_cleverhans,
        Foolbox=bim_foolbox,
        Torchattacks=bim_torchattacks,
    )
    frameworks = [fw for fw in frameworks if fw in fw_func]
    for fw in frameworks:
        start = time.time()
        print(f"Producing BIM adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**bim_params), time.time() - start, fw
    start = time.time()
    yield x + bim.craft(x, y), time.time() - start, "aml"


def bim_advertorch(alpha, clip_max, clip_min, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with BIM in AdverTorch.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: BIM AdverTorch adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from advertorch.attacks import LinfBasicIterativeAttack

    return LinfBasicIterativeAttack(
        predict=model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=epsilon,
        nb_iter=epochs,
        eps_iter=alpha,
        clip_min=clip_min,
        clip_max=clip_max,
        targeted=False,
    ).perturb(x=x.clone(), y=y.clone())


def bim_art(alpha, clip_max, clip_min, epochs, epsilon, model, verbose, x, **_):
    """
    This function crafts adversarial examples with BIM in ART.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :return: BIM ART adversarial examples
    :rtype: torch Tensor object (n, m)
    """

    from art.attacks.evasion import BasicIterativeMethod
    from art.estimators.classification import PyTorchClassifier

    art_classifier = PyTorchClassifier(
        model=model.model,
        clip_values=(clip_min.max().item(), clip_max.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(x.size(1),),
        nb_classes=model.classes,
        device_type="gpu" if model.device == "cuda" else model.device,
    )
    return torch.from_numpy(
        BasicIterativeMethod(
            estimator=art_classifier,
            eps=epsilon,
            eps_step=alpha,
            max_iter=epochs,
            targeted=False,
            batch_size=x.size(0),
            verbose=verbose,
        ).generate(x=x.clone().cpu().numpy())
    ).to(model.device)


def bim_cleverhans(alpha, clip_max, clip_min, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with BIM in CleverHans.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: BIM CleverHans adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from cleverhans.torch.attacks.projected_gradient_descent import (
        projected_gradient_descent as basic_iterative_method,
    )

    return basic_iterative_method(
        model_fn=model,
        x=x.clone(),
        eps=epsilon,
        eps_iter=alpha,
        nb_iter=epochs,
        norm=float("inf"),
        clip_min=clip_min.max(),
        clip_max=clip_max.min(),
        y=y,
        targeted=False,
        rand_init=False,
        rand_minmax=0,
        sanity_checks=False,
    ).detach()


def bim_foolbox(alpha, clip_max, clip_min, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with BIM in Foolbox.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: BIM Foolbox adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from foolbox import PyTorchModel
    from foolbox.attacks import LinfBasicIterativeAttack

    fb_classifier = PyTorchModel(
        model=model.model,
        bounds=(clip_min.min().item(), clip_max.max().item()),
        device=model.device,
    )
    return LinfBasicIterativeAttack(
        rel_stepsize=None,
        abs_stepsize=alpha,
        steps=epochs,
        random_start=False,
    )(fb_classifier, x.clone(), y.clone(), epsilons=epsilon)[1]


def bim_torchattacks(alpha, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with BIM in torchattacks.

    :param alpha: perturbation strength
    :type alpha: float
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: BIM torchattacks adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from torchattacks import BIM

    return BIM(model=model, eps=epsilon, alpha=alpha, steps=epochs)(
        inputs=x.clone(), labels=y
    )


def cwl2(clip_max, clip_min, frameworks, parameters, safe, verbose, x, y):
    """
    This method crafts adversariale examples with CW-L2 (Carlini-Wagner with l2
    norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported frameworks for
    CW-L2 include AdverTorch, ART, CleverHans, Foolbox, and Torchattacks.
    Notably, Torchattacks does not support binary search over c, and, unless
    safeguards are ignored, the CleverHans implementation is skipped as it is
    slow (on the order of days with default parameters on MNIST).

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param safe: whether to skip implementations with high compute or time
    :type safe: bool
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: CW-L2 adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    cwl2 = aml.attacks.cwl2(**parameters)
    cwl2_params = dict(
        alpha=cwl2.alpha,
        c=cwl2.surface.loss.c.item(),
        classes=cwl2.model.classes,
        clip_max=clip_max,
        clip_min=clip_min,
        epochs=cwl2.epochs,
        epsilon=cwl2.epsilon,
        hparam_steps=cwl2.hparam_steps,
        k=cwl2.surface.loss.k,
        model=cwl2.model,
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(
        AdverTorch=cwl2_advertorch,
        ART=cwl2_art,
        CleverHans=cwl2_cleverhans,
        Foolbox=cwl2_foolbox,
        Torchattacks=cwl2_torchattacks,
    )
    frameworks = [fw for fw in frameworks if fw in fw_func]
    safe and "CleverHans" in frameworks and frameworks.remove("CleverHans")
    for fw in frameworks:
        start = time.time()
        print(f"Producing CW-L2 adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**cwl2_params), time.time() - start, fw
    start = time.time()
    yield x + cwl2.craft(x, y), time.time() - start, "aml"


def cwl2_advertorch(
    alpha, c, classes, clip_max, clip_min, epochs, hparam_steps, k, model, x, y, **_
):
    """
    This function crafts adversarial examples with CW-L2 in AdverTorch.

    :param alpha: perturbation strength
    :type alpha: float
    :param c: importance of misclassification over imperceptability
    :type c: float
    :param classes: number of classes
    :type classes: int
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param hparam_steps: number of binary search steps
    :type hpapram_steps: int
    :param k: minimum logit difference between true and current classes
    :type k: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: CW-L2 AdverTorch adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from advertorch.attacks import CarliniWagnerL2Attack

    return CarliniWagnerL2Attack(
        predict=model,
        num_classes=classes,
        confidence=k,
        targeted=False,
        learning_rate=alpha,
        binary_search_steps=hparam_steps,
        max_iterations=epochs,
        abort_early=True,
        initial_const=c,
        clip_min=clip_min,
        clip_max=clip_max,
    ).perturb(x=x.clone(), y=y.clone())


def cwl2_art(
    alpha, c, clip_max, clip_min, epochs, hparam_steps, k, model, verbose, x, **_
):
    """
    This function crafts adversarial examples with CW-L2 in ART.

    :param alpha: perturbation strength
    :type alpha: float
    :param c: importance of misclassification over imperceptability
    :type c: float
    :param classes: number of classes
    :type classes: int
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param hpapram_steps: number of binary search steps
    :type hpapram_steps: int
    :param k: minimum logit difference between true and current classes
    :type k: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :return: CW-L2 ART adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from art.attacks.evasion import CarliniL2Method as CarliniWagner
    from art.estimators.classification import PyTorchClassifier

    art_classifier = PyTorchClassifier(
        model=model.model,
        clip_values=(clip_min.max().item(), clip_max.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(x.size(1),),
        nb_classes=model.classes,
        device_type="gpu" if model.device == "cuda" else model.device,
    )
    return torch.from_numpy(
        CarliniWagner(
            classifier=art_classifier,
            confidence=k,
            targeted=False,
            learning_rate=alpha,
            binary_search_steps=hparam_steps,
            max_iter=epochs,
            initial_const=c,
            max_halving=hparam_steps // 2,
            max_doubling=hparam_steps // 2,
            batch_size=x.size(0),
            verbose=verbose,
        ).generate(x=x.clone().cpu().numpy())
    ).to(model.device)


def cwl2_cleverhans(
    alpha, c, classes, clip_max, clip_min, epochs, hparam_steps, k, model, x, y, **_
):
    """
    This function crafts adversarial examples with CW-L2 in CleverHans.

    :param alpha: perturbation strength
    :type alpha: float
    :param c: importance of misclassification over imperceptability
    :type c: float
    :param classes: number of classes
    :type classes: int
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param hpapram_steps: number of binary search steps
    :type hpapram_steps: int
    :param k: minimum logit difference between true and current classes
    :type k: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: CW-L2 CleverHans adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

    return (
        carlini_wagner_l2(
            model_fn=model,
            x=x.clone(),
            n_classes=classes,
            y=y,
            lr=alpha,
            confidence=k,
            clip_min=clip_min.max(),
            clip_max=clip_max.min(),
            initial_const=c,
            binary_search_steps=hparam_steps,
            max_iterations=epochs,
        ).detach(),
    )


def cwl2_foolbox(
    alpha, c, clip_max, clip_min, epochs, epsilon, hparam_steps, k, model, x, y, **_
):
    """
    This function crafts adversarial examples with CW-L2 in Foolbox.

    :param alpha: perturbation strength
    :type alpha: float
    :param c: importance of misclassification over imperceptability
    :type c: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l2 budget
    :type epsilon: float
    :param hpapram_steps: number of binary search steps
    :type hpapram_steps: int
    :param k: minimum logit difference between true and current classes
    :type k: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: CW-L2 Foolbox adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from foolbox import PyTorchModel
    from foolbox.attacks import L2CarliniWagnerAttack

    fb_classifier = PyTorchModel(
        model=model.model,
        bounds=(clip_min.min().item(), clip_max.max().item()),
        device=model.device,
    )
    return L2CarliniWagnerAttack(
        binary_search_steps=hparam_steps,
        steps=epochs,
        stepsize=alpha,
        confidence=k,
        initial_const=c,
        abort_early=True,
    )(fb_classifier, x.clone(), y.clone(), epsilons=epsilon)[1]


def cwl2_torchattacks(alpha, c, epochs, k, model, x, y, **_):
    """
    This function crafts adversarial examples with CW-L2 in torchattacks.
    Notably, this implementation can return nans.

    :param alpha: perturbation strength
    :type alpha: float
    :param c: importance of misclassification over imperceptability
    :type c: float
    :param epochs: number of attack iterations
    :type epochs: int
    :param k: minimum logit difference between true and current classes
    :type k: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: CW-L2 torchattacks adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from torchattacks import CW

    adv = CW(model=model, c=c, kappa=k, steps=epochs, lr=alpha)(
        inputs=x.clone(), labels=y
    )
    return torch.where(adv.isnan(), x, adv)


def df(clip_max, clip_min, frameworks, parameters, safe, verbose, x, y):
    """
    This method crafts adversarial examples with DF (DeepFool)
    (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
    include ART, Foolbox, and Torchattacks. Notably, the DF implementation in
    ART, Foolbox, and Torchattacks have an overshoot parameter which we set to
    aml DF's learning rate alpha minus one (not to be confused with the epsilon
    parameter used in aml, which governs the norm-ball size). Finally, the
    Torchattacks implementation is skipped if the number of classes is greater
    than 4, as it has a slow runtime for computing model Jacobians.

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param safe: whether to skip implementations with high compute or time
    :type safe: bool
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: DF adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    df = aml.attacks.df(**parameters)
    df_params = dict(
        classes=df.model.classes,
        clip_max=clip_max,
        clip_min=clip_min,
        epochs=df.epochs,
        epsilon=df.epsilon,
        overshoot=df.alpha - 1,
        model=df.model,
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(ART=df_art, Foolbox=df_foolbox, Torchattacks=df_torchattacks)
    frameworks = [fw for fw in frameworks if fw in fw_func]
    safe and "Torchattacks" in frameworks and df_params[
        "classes"
    ] > 4 and frameworks.remove("Torchattacks")
    for fw in frameworks:
        start = time.time()
        print(f"Producing DF adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**df_params), time.time() - start, fw
    start = time.time()
    yield x + df.craft(x, y), time.time() - start, "aml"


def df_art(classes, clip_max, clip_min, epochs, overshoot, model, verbose, x, **_):
    """
    This function crafts adversarial examples with DF in ART.

    :param classes: number of classes
    :type classes: int
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param overshoot: additional perturbation amount per iteration
    :type overshoot: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :return: DF ART adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from art.attacks.evasion import DeepFool
    from art.estimators.classification import PyTorchClassifier

    art_classifier = PyTorchClassifier(
        model=model.model,
        clip_values=(clip_min.max().item(), clip_max.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(x.size(1),),
        nb_classes=model.classes,
        device_type="gpu" if model.device == "cuda" else model.device,
    )
    return torch.from_numpy(
        DeepFool(
            classifier=art_classifier,
            max_iter=epochs,
            epsilon=overshoot,
            nb_grads=classes,
            batch_size=x.size(0),
            verbose=verbose,
        ).generate(x=x.clone().cpu().numpy())
    ).to(model.device)


def df_foolbox(
    classes, clip_max, clip_min, epochs, epsilon, model, overshoot, x, y, **_
):
    """
    This function crafts adversarial examples with DF in Foolbox.

    :param classes: number of classes
    :type classes: int
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l2 budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param overshoot: additional perturbation amount per iteration
    :type overshoot: float
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: DF Foolbox adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from foolbox import PyTorchModel
    from foolbox.attacks import L2DeepFoolAttack as DeepFoolAttack

    fb_classifier = PyTorchModel(
        model=model.model,
        bounds=(clip_min.min().item(), clip_max.max().item()),
        device=model.device,
    )
    return DeepFoolAttack(
        steps=epochs,
        candidates=classes,
        overshoot=overshoot,
        loss="logits",
    )(fb_classifier, x.clone(), y.clone(), epsilons=epsilon)[1]


def df_torchattacks(epochs, model, overshoot, x, y, **_):
    """
    This function crafts adversarial examples with DF in torchattacks.
    Notably, this implementation can return nans.

    :param alpha: perturbation strength
    :type alpha: float
    :param epochs: number of attack iterations
    :type epochs: int
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param overshoot: additional perturbation amount per iteration
    :type overshoot: float
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: DF torchattacks adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from torchattacks import DeepFool

    adv = DeepFool(model=model, steps=epochs, overshoot=overshoot)(
        inputs=x.clone(), labels=y
    )
    return torch.where(adv.isnan(), x, adv)


def fab(frameworks, parameters, verbose, x, y, **_):
    """
    This method crafts adversarial examples with FAB (Fast Adaptive Boundary)
    (https://arxiv.org/pdf/1907.02044.pdf). The supported frameworks for FAB
    include AdverTorch and Torchattacks.

    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: DF adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    fab = aml.attacks.fab(**parameters)
    fab_params = dict(
        alpha=fab.alpha,
        alpha_max=fab.traveler.optimizer.param_groups[0]["alpha_max"],
        beta=fab.traveler.optimizer.param_groups[0]["beta"],
        classes=fab.model.classes,
        epochs=fab.epochs,
        epsilon=fab.epsilon,
        num_restarts=fab.num_restarts,
        model=fab.model,
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(Advertorch=fab_advertorch, Torchattacks=fab_torchattacks)
    frameworks = [fw for fw in frameworks if fw in fw_func]
    for fw in frameworks:
        reset_seeds()
        start = time.time()
        print(f"Producing FAB adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**fab_params), time.time() - start, fw
    reset_seeds()
    start = time.time()
    yield x + fab.craft(x, y), time.time() - start, "aml"


def fab_advertorch(
    alpha, alpha_max, beta, epochs, epsilon, model, num_restarts, verbose, x, y
):
    """
    This function crafts adversarial examples with FAB in AdverTorch.

    :param alpha: perturbation strength
    :type alpha: float
    :param alpha_max: maximum value of alpha
    :type alpha_max: float
    :param beta: backward step strength
    :type beta: float
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l2 budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of random restarts to perform
    :type num_restarts: int
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: FAB AdverTorch adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from advertorch.attacks import FABAttack

    return FABAttack(
        predict=model,
        norm="L2",
        n_restarts=num_restarts,
        n_iter=epochs,
        eps=epsilon,
        alpha_max=alpha_max,
        eta=alpha,
        beta=beta,
        verbose=verbose,
    ).perturb(x=x.clone(), y=y.clone())


def fab_torchattacks(
    alpha, alpha_max, beta, classes, epochs, epsilon, model, num_restarts, verbose, x, y
):
    """
    This function crafts adversarial examples with fab in torchattacks.

    :param alpha: perturbation strength
    :type alpha: float
    :param alpha_max: maximum value of alpha
    :type alpha_max: float
    :param beta: backward step strength
    :type beta: float
    :param classes: number of classes
    :type classes: int
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of random restarts to perform
    :type num_restarts: int
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: DF torchattacks adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from torchattacks import FAB

    return FAB(
        model=model,
        norm="L2",
        eps=epsilon,
        steps=epochs,
        n_restarts=num_restarts,
        alpha_max=alpha_max,
        eta=alpha,
        beta=beta,
        verbose=verbose,
        seed=0,
        multi_targeted=False,
        n_classes=classes,
    )(inputs=x.clone(), labels=y)


def jsma(clip_max, clip_min, frameworks, parameters, safe, verbose, x, y):
    """
    This method crafts adversarial examples with JSMA (Jacobian Saliency Map
    Approach) (https://arxiv.org/pdf/1511.07528.pdf). The supported frameworks
    for the JSMA include AdverTorch and ART. Notably, the JSMA implementation
    in AdverTorch and ART both assume the l0-norm is passed in as a percentage
    and we set theta to be 1 since features can only be perturbed once.
    moreover, the advertorch implementation: (1) does not suport an untargetted
    scheme (so we supply random targets), and (2) computes every pixel pair at
    once, often leading to memory crashes (e.g., it'll terminate on a 16GB
    system with MNIST), so this attack is skipped when the feature space is
    greater than 784.

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param safe: whether to skip implementations with high compute or time
    :type safe: bool
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: DF adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    jsma = aml.attacks.jsma(**parameters | dict(alpha=1))
    jsma_params = dict(
        alpha=jsma.alpha,
        classes=jsma.model.classes,
        clip_max=clip_max,
        clip_min=clip_min,
        epsilon=jsma.epsilon / x.size(1),
        model=jsma.model,
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(Advertorch=jsma_advertorch, Art=jsma_art)
    frameworks = [fw for fw in frameworks if fw in fw_func]
    safe and "AdverTorch" in frameworks and x.size(1) >= 784 and frameworks.remove(
        "AdverTorch"
    )
    for fw in frameworks:
        start = time.time()
        print(f"Producing JSMA adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**jsma_params), time.time() - start, fw
    start = time.time()
    yield x + jsma.craft(x, y), time.time() - start, "aml"


def jsma_advertorch(alpha, classes, clip_max, clip_min, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with JSMA in AdverTorch.

    :param alpha: perturbation strength
    :type alpha: float
    :param classes: number of classes
    :type classes: int
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epsilon: l0 budget (as a percentage of the feature space)
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: JSMA AdverTorch adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from advertorch.attacks import JacobianSaliencyMapAttack

    return JacobianSaliencyMapAttack(
        predict=model,
        num_classes=classes,
        clip_min=clip_min,
        clip_max=clip_max,
        gamma=epsilon,
        theta=alpha,
    ).perturb(x=x.clone(), y=torch.randint(classes, y.size(), device=model.device))


def jsma_art(alpha, clip_max, clip_min, epsilon, model, verbose, x, **_):
    """
    This function crafts adversarial examples with JSMA in ART.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epsilon: l0 budget (as a percentage of the feature space)
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :return: JSMA ART adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from art.attacks.evasion import SaliencyMapMethod
    from art.estimators.classification import PyTorchClassifier

    art_classifier = PyTorchClassifier(
        model=model.model,
        clip_values=(clip_min.max().item(), clip_max.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(x.size(1),),
        nb_classes=model.classes,
        device_type="gpu" if model.device == "cuda" else model.device,
    )
    return torch.from_numpy(
        SaliencyMapMethod(
            classifier=art_classifier,
            theta=alpha,
            gamma=epsilon,
            batch_size=x.size(0),
            verbose=verbose,
        ).generate(x=x.clone().cpu().numpy())
    ).to(model.device)


def l0_proj(l0, p):
    """
    This method projects perturbation vectors so that they are complaint with
    the specified l0 threat model. Specifically, the components of
    perturbations that exceed the threat model are set to zero, sorted by
    increasing magnitude.

    :param l0: l0 threat model
    :type l0: int
    :param p: perturbation vectors
    :type p: torch Tensor object (n, m)
    :return: threat-model-compliant perturbation vectors
    :rtype: torch Tensor object (n, m)
    """
    return p.scatter(1, p.abs().sort(1).indices[:, : p.size(1) - l0], 0)


def l2_proj(l2, p):
    """
    This method projects perturbation vectors so that they are complaint with
    the specified l2 threat model Specifically, perturbation vectors whose
    l2-norms exceed the threat model are normalized by their l2-norms times
    epsilon.

    :param l2: l2 threat model
    :type l2: float
    :param p: perturbation vectors
    :type p: torch Tensor object (n, m)
    :return: threat-model-compliant perturbation vectors
    :rtype: torch Tensor object (n, m)
    """
    return p.renorm(2, 0, l2)


def linf_proj(linf, p):
    """
    This method projects perturbation vectors so that they are complaint with
    the specified l∞ threat model. Specifically, perturbation vectors whose
    l∞-norms exceed the threat model are are clipped to ±epsilon.

    :param linf: l∞ threat model
    :type linf: float
    :param p: perturbation vectors
    :type p: torch Tensor object (n, m)
    :return: threat-model-compliant perturbation vectors
    :rtype: torch Tensor object (n, m)
    """
    return p.clamp(-linf, linf)


def main(
    alpha,
    attacks,
    budget,
    dataset,
    device,
    epochs,
    frameworks,
    safe,
    trials,
    verbose,
):
    """
    This function is the main entry point for the framework comparison
    benchmark. Specifically this: (1) loads data, (2) trains a model, (3)
    computes attack parameters, (4) crafts adversarial examples for each
    framework, (5) measures attack statistics, and (6) plots the results.

    :param alpha: perturbation strength, per-iteration
    :type alpha: float
    :param attacks: attacks to use
    :type attacks: list of functions
    :param budget: maximum lp budget
    :type budget: float
    :param dataset: dataset to use
    :type dataset: tuple of str
    :param device: hardware device to use
    :param device: str
    :param epochs: number of attack iterations
    :type epochs: int
    :param frameworks: frameworks to compare against
    :type frameworks: tuple of str
    :param safe: ignore attacks that have high compute or time costs
    :type safe: bool
    :param trials: number of experiment trials
    :type trials: int
    :param verbose: enable verbose output
    :type verbose: bool
    :return: None
    :rtype: NoneType
    """
    print(
        f"Using {len(attacks)} attacks on {dataset} with "
        f"{len(frameworks)} frameworks in {trials} trials..."
    )

    # load dataset and determine clips (non-image datasets may not be 0-1)
    data = getattr(mlds, dataset)
    try:
        xt = torch.from_numpy(data.train.data).to(device)
        yt = torch.from_numpy(data.train.labels).long().to(device)
        x = torch.from_numpy(data.test.data).to(device)
        y = torch.from_numpy(data.test.labels).long().to(device)
    except AttributeError:
        xt = torch.from_numpy(data.dataset.data).to(device)
        yt = torch.from_numpy(data.dataset.labels).long().to(device)
        x = xt
        y = yt
    cmin, cmax = x.min(0).values.clamp(max=0), x.max(0).values.clamp(min=1)

    # load model hyperparameters and architecture
    params = dict(auto_batch=0, device=device, verbosity=0.25 if verbose else 0)
    template = getattr(dlm.templates, dataset)
    model = (
        dlm.CNNClassifier(**template.cnn | params)
        if hasattr(template, "cnn")
        else dlm.MLPClassifier(**template.mlp | params)
    )

    # compute attack budgets and set base attack parameters
    norms = collections.namedtuple("norms", ("budget", "projection", "ord"))
    l0 = int(x.size(1) * budget) + 1
    l2 = torch.stack((cmin, cmax)).diff(dim=0).norm(2).item() * budget
    linf = budget
    norms = dict(
        apgdce=norms(linf, linf_proj, torch.inf),
        apgddlr=norms(linf, linf_proj, torch.inf),
        bim=norms(linf, linf_proj, torch.inf),
        cwl2=norms(l2, l2_proj, 2),
        df=norms(l2, l2_proj, 2),
        fab=norms(l2, l2_proj, 2),
        jsma=norms(l0, l0_proj, 0),
        pgd=norms(linf, linf_proj, torch.inf),
    )
    params = dict(
        clip_max=cmax,
        clip_min=cmin,
        frameworks=frameworks,
        safe=safe,
        verbose=verbose,
        x=x,
        y=y,
    )
    attack_params = dict(
        alpha=alpha, epochs=epochs, model=model, verbosity=float(verbose)
    )

    # initialize results dataframe, configure general parameters, and set verbosity
    metrics = "attack", "baseline", "framework", "accuracy", "budget", "time"
    results = pandas.DataFrame(columns=metrics)
    end = "\n" if verbose else "\r"
    row = 0

    # train model and compute accuracy baseline
    for t in range(1, trials + 1):
        print(f"Training {dataset} model... Trial {t} of {trials}")
        model.fit(xt, yt)
        test_acc = model.accuracy(x, y).item() * 100

        # craft adversarial examples & ensure they comply with clips and budget
        for j, attack in enumerate(attacks):
            print(f"Attacking with {attack.__name__}... {j} of {len(attacks)}", end=end)
            attack_params.update(dict(epsilon=norms[attack].budget))
            for adv, times, fw in attack(**params | dict(parameters=attack_params)):
                print(f"Computing results for {fw} {attack.__name__}...", end=end)
                budget = norms[attack].budget
                norm = norms[attack].ord
                adv = adv.clamp(cmin, cmax)
                p = norms[attack].projection(norms[attack].budget, adv.sub(x))
                acc = model.accuracy(x + p, y).mul(100).item()
                used = p.norm(norm, 1).mean().div(budget).mul(100).item()
                results.loc[row] = attack.__name__, test_acc, fw, acc, used, times
                row += 1

    # compute median, plot results, and save
    plot(dataset, results)
    return None


def plot(dataset, results):
    """
    This function plots the framework comparsion results. Specifically, this
    produces a scatter plot containing the model accuracy on the crafting set
    over the percentage of the budget consumed (with a dotted line designating
    the original model accuracy on the clean data) with axes in log scale to
    show separation. A grouped bar chat is also paired with the scatter plot
    containing attacks over the crafting time (in seconds or minutes) with
    error bars representing the standard deviation. Frameworks are divided by
    color and attacks by marker style. The plot is written to disk in the
    current directory.

    :param dataset: dataset used
    :type dataset: str
    :param results: results of the framework comparison
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """

    # take the median of the crafting results, scatter, & add a refline
    fig, axes = plt.subplots(1, 2, layout="constrained", subplot_kw=dict(box_aspect=1))
    seaborn.scatterplot(
        alpha=0.6,
        ax=axes[0],
        clip_on=False,
        data=results.groupby(["attack", "framework"]).median().reset_index(),
        hue="framework",
        s=100,
        style="attack",
        x="budget",
        y="accuracy",
    )
    axes[0].axhline(color="r", label="baseline", linestyle="--", y=results.baseline[0])
    axes[0].set(xlabel="budget consumed", xscale="symlog", yscale="symlog")
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    # add the bar chart to the next subplot and scale time if necessary
    use_minutes = results.time.max() > 120
    results.time = results.time / 60 if use_minutes else results.time
    seaborn.barplot(
        data=results,
        errorbar="sd",
        errwidth=0.5,
        hue="framework",
        x="time",
        y="attack",
    )
    axes[1].grid(axis="x")
    axes[1].set(
        axisbelow=True, xlabel="minutes" if use_minutes else "seconds", xscale="log"
    )
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # configure legend (ensure frameworks and attacks are separate) and save
    seaborn.despine()
    handles, labels = axes[0].get_legend_handles_labels()
    idx = atk if (atk := labels.index("attack")) < len(labels) / 2 else len(labels)
    handles = handles[:idx] + [handles[0]] * abs(atk * 2 - len(handles)) + handles[idx:]
    labels = labels[:idx] + [""] * abs(atk * 2 - len(labels)) + labels[idx:]
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.legend(
        bbox_to_anchor=(0.98, 0.75),
        frameon=False,
        handles=handles,
        labels=labels,
        loc="upper left",
        ncols=2,
    )
    fig.suptitle(f"dataset={dataset}", y=0.80)
    fig.savefig(__file__[:-3] + f"_{dataset}.pdf", bbox_inches="tight")
    return None


def pgd(clip_max, clip_min, frameworks, parameters, verbose, x, y, **_):
    """
    This method crafts adversarial examples with PGD (Projected Gradient
    Descent)) (https://arxiv.org/pdf/1706.06083.pdf). The supported frameworks
    for PGD include AdverTorch, ART, CleverHans, and Torchattacks.

    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: PGD adversarial examples
    :rtype: generator of tuples of torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    pgd = aml.attacks.pgd(**parameters)
    pgd_params = dict(
        alpha=pgd.alpha,
        clip_max=clip_max,
        clip_min=clip_min,
        epochs=pgd.epochs,
        epsilon=pgd.epsilon,
        model=pgd.model,
        verbose=verbose,
        x=x,
        y=y,
    )
    fw_func = dict(
        AdverTorch=pgd_advertorch,
        ART=pgd_art,
        CleverHans=pgd_cleverhans,
        Foolbox=pgd_foolbox,
        Torchattacks=pgd_torchattacks,
    )
    frameworks = [fw for fw in frameworks if fw in fw_func]
    for fw in frameworks:
        reset_seeds()
        start = time.time()
        print(f"Producing PGD adversarial examples with {fw}...", end=end)
        yield fw_func[fw](**pgd_params), time.time() - start, fw
    reset_seeds()
    start = time.time()
    yield x + pgd.craft(x, y), time.time() - start, "aml"


def pgd_advertorch(alpha, clip_max, clip_min, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with PGD in AdverTorch.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarial examples from
    :type y: torch Tensor object (n,)
    :return: PGD AdverTorch adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from advertorch.attacks import PGDAttack

    return PGDAttack(
        predict=model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=epsilon,
        nb_iter=epochs,
        eps_iter=alpha,
        rand_init=True,
        clip_min=clip_min,
        clip_max=clip_max,
        ord=float("inf"),
        targeted=False,
    ).perturb(x=x.clone(), y=y.clone())


def pgd_art(alpha, clip_max, clip_min, epochs, epsilon, model, verbose, x, **_):
    """
    This function crafts adversarial examples with PGD in ART.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param verbose: enable verbose output
    :type verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :return: PGD ART adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier

    art_classifier = PyTorchClassifier(
        model=model.model,
        clip_values=(clip_min.max().item(), clip_max.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(x.size(1),),
        nb_classes=model.classes,
        device_type="gpu" if model.device == "cuda" else model.device,
    )
    return torch.from_numpy(
        ProjectedGradientDescent(
            estimator=art_classifier,
            norm="inf",
            eps=epsilon,
            eps_step=alpha,
            decay=None,
            max_iter=epochs,
            targeted=False,
            num_random_init=1,
            batch_size=x.size(0),
            random_eps=True,
            summary_writer=False,
            verbose=verbose,
        ).generate(x=x.clone().cpu().numpy())
    ).to(model.device)


def pgd_cleverhans(alpha, clip_max, clip_min, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with PGD in CleverHans.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: PGD CleverHans adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from cleverhans.torch.attacks.projected_gradient_descent import (
        projected_gradient_descent,
    )

    return projected_gradient_descent(
        model_fn=model,
        x=x.clone(),
        eps=epsilon,
        eps_iter=alpha,
        nb_iter=epochs,
        norm=float("inf"),
        clip_min=clip_min.max(),
        clip_max=clip_max.min(),
        y=y,
        targeted=False,
        rand_init=True,
        rand_minmax=0,
        sanity_checks=False,
    ).detach()


def pgd_foolbox(alpha, clip_max, clip_min, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with PGD in Foolbox.

    :param alpha: perturbation strength
    :type alpha: float
    :param clip_max: maximum feature ranges
    :type clip_max: torch Tensor object (m,)
    :param clip_min: minimum feature ranges
    :type clip_min: torch Tensor object (m,)
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: PGD Foolbox adversarial examples
    :rtype: torch Tensor object (n, m)
    """

    from foolbox import PyTorchModel
    from foolbox.attacks import LinfProjectedGradientDescentAttack

    fb_classifier = PyTorchModel(
        model=model.model,
        bounds=(clip_min.min().item(), clip_max.max().item()),
        device=model.device,
    )
    return LinfProjectedGradientDescentAttack(
        rel_stepsize=None,
        abs_stepsize=alpha,
        steps=epochs,
        random_start=True,
    )(fb_classifier, x.clone(), y.clone(), epsilons=epsilon)[1]


def pgd_torchattacks(alpha, epochs, epsilon, model, x, y, **_):
    """
    This function crafts adversarial examples with PGD in torchattacks.

    :param alpha: perturbation strength
    :type alpha: float
    :param epochs: number of attack iterations
    :type epochs: int
    :param epsilon: l∞ budget
    :type epsilon: float
    :param model: model to craft adversarial examples with
    :type model: dlm LinearClassifier-inherited object
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type y: torch Tensor object (n,)
    :return: PGD torchattacks adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    from torchattacks import PGD

    return PGD(model=model, eps=epsilon, alpha=alpha, steps=epochs, random_start=True)(
        inputs=x.clone(), labels=y
    )


def print(*args, **kwargs):
    """
    This function overrides the print builtin by prepending a timestamp all
    print calls.

    :param *args: position arguments supported by builtin.print
    :type *args: tuple
    :param **kwargs: keyword arguments supported by builtin.print
    :type **kwargs: dictionary
    :return: None
    :rtype: NoneType
    """
    builtins.print(f"[{time.asctime()}]", *args, **kwargs)
    return None


def reset_seeds():
    """
    This function resets the seeds for random number generators used in attacks
    with randomized components.

    :return: None
    :rtype: NoneType
    """
    torch.manual_seed(0)
    np.random.seed(0)
    return None


if __name__ == "__main__":
    """
    This script benchmarks the performance of the aml library against other
    popular adversarial machine learning libraries. Datasets are provided by
    mlds (https://github.com/sheatsley/datasets), models by dlm
    (https://github.com/sheatsley/models), and the following frameworks are
    compared against: AdverTorch (https://github.com/BorealisAI/advertorch),
    ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox),
    CleverHans (https://github.com/cleverhans-lab/cleverhans), Foolbox
    (https://github.com/bethgelab/foolbox), and Torchattacks
    (https://github.com/Harry24k/adversarial-attacks-pytorch). Specifically,
    this script: (1) parses command-line arguments, (2) loads a dataset, (3)
    trains a model, (4) crafts adversarial examples with each framework, (5)
    collects statistics on model accuracy, lp-norm, and crafting time of the
    adversarial examples, and (6) plots the results.
    """

    # determine available frameworks
    frameworks = ("AdverTorch", "ART", "CleverHans", "Foolbox", "Torchattacks")
    frameworks_avail = [f for f in frameworks if importlib.util.find_spec(f.lower())]

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Adversarial machine learning framework comparison"
    )
    parser.add_argument(
        "--alpha",
        default=0.01,
        help="Perturbation strength, per-iteration",
        type=float,
    )
    parser.add_argument(
        "-a",
        "--attacks",
        choices=(apgdce, apgddlr, bim, cwl2, df, fab, jsma, pgd),
        default=(apgdce, apgddlr, bim, cwl2, df, fab, jsma, pgd),
        help="Attacks to use",
        nargs="+",
        type=lambda a: globals()[a],
    )
    parser.add_argument(
        "-b",
        "--budget",
        default=0.15,
        help="Maximum lp budget (as a percent)",
        type=float,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=mlds.__available__,
        default="phishing",
        help="Dataset to use",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Hardware device to use",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        help="Number of attack iterations",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        choices=frameworks_avail,
        default=frameworks_avail,
        help="Frameworks to compare against",
        nargs="*",
        type=lambda f: {f.lower(): f for f in frameworks_avail}[f.lower()],
    )
    parser.add_argument(
        "-i",
        "--ignore-safegaurds",
        action="store_true",
        help="Forcibly run specified attacks/frameworks (*very* time & compute heavy!)",
    ),
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        help="Number of experiment trials",
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output)",
    )
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        attacks=args.attacks,
        budget=args.budget,
        dataset=args.dataset,
        device=args.device,
        epochs=args.epochs,
        frameworks=tuple(args.frameworks),
        safe=not args.ignore_safegaurds,
        trials=args.trials,
        verbose=args.verbose,
    )
    raise SystemExit(0)
