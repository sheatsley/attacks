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
import pickle
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


def apgdce(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with APGD-CE (Auto-PGD with CE
    loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks for
    APGD-CE include ART and Torchattacks. Notably, the Torchattacks
    implementation assumes an image (batches, channels, width, height).

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: APGD-CE adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    apgdce = aml.attacks.apgdce(**parameters)
    model, eps, eps_step, max_iter, nb_random_init, rho = (
        apgdce.model,
        apgdce.epsilon,
        apgdce.alpha,
        apgdce.epochs,
        apgdce.num_restarts,
        apgdce.traveler.optimizer.param_groups[0]["rho"],
    )
    art_adv = ta_adv = None
    reset_seeds()
    start = time.time()
    aml_adv = (x + apgdce.craft(x, y), time.time() - start, "aml")
    if "art" in frameworks:
        from art.attacks.evasion import AutoProjectedGradientDescent

        print("Producing APGD-CE adversarial examples with ART...", end=end)
        reset_seeds()
        start = time.time()
        art_adv = (
            torch.from_numpy(
                AutoProjectedGradientDescent(
                    estimator=art_classifier,
                    norm="inf",
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=max_iter,
                    targeted=False,
                    nb_random_init=nb_random_init,
                    batch_size=x.size(0),
                    loss_type="cross_entropy",
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    if "torchattacks" in frameworks and hasattr(model, "shape"):
        from torchattacks import APGD

        print("Producing APGD-CE adversarial examples with Torchattacks...", end=end)
        reset_seeds()
        ta_x = x.clone().unflatten(1, model.shape)
        start = time.time()
        ta_adv = (
            APGD(
                model=model,
                norm="Linf",
                eps=eps,
                steps=max_iter,
                n_restarts=nb_random_init,
                seed=0,
                loss="ce",
                eot_iter=1,
                rho=rho,
                verbose=verbose,
            )(inputs=ta_x, labels=y)
            .flatten(1)
            .detach(),
            time.time() - start,
            "Torchattacks",
        )
    reset_seeds()
    return tuple([fw for fw in (art_adv, ta_adv) if fw is not None] + [aml_adv])


def apgddlr(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with APGD-DLR (Auto-PGD with DLR
    loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks for
    APGD-DLR include ART and Torchattacks. Notably, DLR loss is undefined for
    these frameworks when there are only two classes. Moreover, the
    Torchattacks implementation assumes an image (batches, channels, width,
    height).

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: APGD-DLR adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    apgddlr = aml.attacks.apgddlr(**parameters)
    model, eps, eps_step, max_iter, nb_random_init, rho = (
        apgddlr.model,
        apgddlr.epsilon,
        apgddlr.alpha,
        apgddlr.epochs,
        apgddlr.num_restarts,
        apgddlr.traveler.optimizer.param_groups[0]["rho"],
    )
    art_adv = ta_adv = None
    reset_seeds()
    start = time.time()
    aml_adv = (x + apgddlr.craft(x, y), time.time() - start, "aml")
    if "art" in frameworks and art_classifier.nb_classes > 2:
        from art.attacks.evasion import AutoProjectedGradientDescent

        print("Producing APGD-DLR adversarial examples with ART...", end=end)
        reset_seeds()
        start = time.time()
        art_adv = (
            torch.from_numpy(
                AutoProjectedGradientDescent(
                    estimator=art_classifier,
                    norm="inf",
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=max_iter,
                    targeted=False,
                    nb_random_init=nb_random_init,
                    batch_size=x.size(0),
                    loss_type="difference_logits_ratio",
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    if (
        "torchattacks" in frameworks
        and art_classifier.nb_classes > 2
        and hasattr(model, "shape")
    ):
        from torchattacks import APGD

        print("Producing APGD-DLR adversarial examples with Torchattacks...", end=end)
        reset_seeds()
        ta_x = x.clone().unflatten(1, model.shape)
        start = time.time()
        ta_adv = (
            APGD(
                model=model,
                norm="Linf",
                eps=eps,
                steps=max_iter,
                n_restarts=nb_random_init,
                seed=0,
                loss="dlr",
                eot_iter=1,
                rho=rho,
                verbose=verbose,
            )(inputs=ta_x, labels=y)
            .flatten(1)
            .detach(),
            time.time() - start,
            "Torchattacks",
        )
    reset_seeds()
    return tuple([fw for fw in (art_adv, ta_adv) if fw is not None] + [aml_adv])


def bim(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with BIM (Basic Iterative
    Method) (https://arxiv.org/pdf/1611.01236.pdf). The supported
    frameworks for BIM include AdverTorch, ART, CleverHans, Foolbox, and
    Torchattacks. Notably, CleverHans does not have an explicit
    implementation of BIM, so we call PGD, but with random initialization
    disabled.

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: BIM adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    clip_min, clip_max = clip.unbind()
    bim = aml.attacks.bim(**parameters)
    model, eps, nb_iter, eps_iter = bim.model, bim.epsilon, bim.epochs, bim.alpha
    at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
    start = time.time()
    aml_adv = (x + bim.craft(x, y), time.time() - start, "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import LinfBasicIterativeAttack as BasicIterativeAttack

        print("Producing BIM adversarial examples with AdverTorch...", end=end)
        start = time.time()
        at_adv = (
            BasicIterativeAttack(
                predict=model,
                loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                eps=eps,
                nb_iter=nb_iter,
                eps_iter=eps_iter,
                clip_min=clip_min,
                clip_max=clip_max,
                targeted=False,
            ).perturb(x=x.clone(), y=y.clone()),
            time.time() - start,
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import BasicIterativeMethod

        print("Producing BIM adversarial examples with ART...", end=end)
        start = time.time()
        art_adv = (
            torch.from_numpy(
                BasicIterativeMethod(
                    estimator=art_classifier,
                    eps=eps,
                    eps_step=eps_iter,
                    max_iter=nb_iter,
                    targeted=False,
                    batch_size=x.size(0),
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    if "cleverhans" in frameworks:
        from cleverhans.torch.attacks.projected_gradient_descent import (
            projected_gradient_descent as basic_iterative_method,
        )

        print("Producing BIM adversarial examples with CleverHans...", end=end)
        start = time.time()
        ch_adv = (
            basic_iterative_method(
                model_fn=model,
                x=x.clone(),
                eps=eps,
                eps_iter=eps_iter,
                nb_iter=nb_iter,
                norm=float("inf"),
                clip_min=clip_min.max(),
                clip_max=clip_max.min(),
                y=y,
                targeted=False,
                rand_init=False,
                rand_minmax=0,
                sanity_checks=False,
            ).detach(),
            time.time() - start,
            "CleverHans",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import LinfBasicIterativeAttack as BasicIterativeAttack

        print("Producing BIM adversarial examples with Foolbox...", end=end)
        start = time.time()
        _, fb_adv, _ = BasicIterativeAttack(
            rel_stepsize=None,
            abs_stepsize=eps_iter,
            steps=nb_iter,
            random_start=False,
        )(fb_classifier, x.clone(), y.clone(), epsilons=eps)
        fb_adv = (fb_adv, time.time() - start, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import BIM

        print("Producing BIM adversarial examples with Torchattacks...", end=end)
        start = time.time()
        ta_adv = (
            BIM(model=model, eps=eps, alpha=eps_iter, steps=nb_iter)(
                inputs=x.clone(), labels=y
            ),
            time.time() - start,
            "Torchattacks",
        )
    return tuple(
        [fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None]
        + [aml_adv]
    )


def cwl2(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversariale examples with CW-L2 (Carlini-Wagner with l2
    norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported frameworks for
    CW-L2 include AdverTorch, ART, CleverHans, Foolbox, and Torchattacks.
    Notably, Torchattacks does not explicitly support binary searching on c (it
    expects searching manually). Notably, while our implementation shows
    competitive performance with low iterations, other implementations (sans
    ART) require a substantial number of additional iterations, so the minimum
    number of steps is set to be at least 300.

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: CW-L2 adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    clip_min, clip_max = clip.unbind()
    cwl2 = aml.attacks.cwl2(**parameters)
    at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
    (
        model,
        classes,
        confidence,
        learning_rate,
        binary_search_steps,
        max_iterations,
        initial_const,
    ) = (
        cwl2.model,
        cwl2.model.classes,
        cwl2.surface.loss.k,
        cwl2.alpha,
        cwl2.hparam_steps,
        max(cwl2.epochs, 300),
        cwl2.surface.loss.c.item(),
    )
    at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
    start = time.time()
    aml_adv = (x + cwl2.craft(x, y), time.time() - start, "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import CarliniWagnerL2Attack

        print("Producing CW-L2 adversarial examples with AdverTorch...", end=end)
        start = time.time()
        at_adv = (
            CarliniWagnerL2Attack(
                predict=model,
                num_classes=classes,
                confidence=confidence,
                targeted=False,
                learning_rate=learning_rate,
                binary_search_steps=binary_search_steps,
                max_iterations=max_iterations,
                abort_early=True,
                initial_const=initial_const,
                clip_min=clip_min,
                clip_max=clip_max,
            ).perturb(x=x.clone(), y=y.clone()),
            time.time() - start,
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import CarliniL2Method as CarliniWagner

        print("Producing CW-L2 adversarial examples with ART...", end=end)
        start = time.time()
        art_adv = (
            torch.from_numpy(
                CarliniWagner(
                    classifier=art_classifier,
                    confidence=confidence,
                    targeted=False,
                    learning_rate=learning_rate,
                    binary_search_steps=binary_search_steps,
                    max_iter=max_iterations,
                    initial_const=initial_const,
                    max_halving=binary_search_steps // 2,
                    max_doubling=binary_search_steps // 2,
                    batch_size=x.size(0),
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    if "cleverhans" in frameworks:
        from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

        print("Producing CW-L2 adversarial examples with CleverHans...", end=end)
        start = time.time()
        ch_adv = (
            carlini_wagner_l2(
                model_fn=model,
                x=x.clone(),
                n_classes=classes,
                y=y,
                lr=learning_rate,
                confidence=confidence,
                clip_min=clip_min.max(),
                clip_max=clip_max.min(),
                initial_const=initial_const,
                binary_search_steps=binary_search_steps,
                max_iterations=max_iterations,
            ).detach(),
            time.time() - start,
            "CleverHans",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import L2CarliniWagnerAttack

        print("Producing CW-L2 adversarial examples with Foolbox...", end=end)
        start = time.time()
        _, fb_adv, _ = L2CarliniWagnerAttack(
            binary_search_steps=binary_search_steps,
            steps=max_iterations,
            stepsize=learning_rate,
            confidence=confidence,
            initial_const=initial_const,
            abort_early=True,
        )(fb_classifier, x.clone(), y.clone(), epsilons=cwl2.epsilon)
        fb_adv = (fb_adv, time.time() - start, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import CW

        print("Producing CW-L2 adversarial examples with Torchattacks...", end=end)
        start = time.time()
        ta_adv = CW(
            model=model,
            c=initial_const,
            kappa=confidence,
            steps=max_iterations,
            lr=learning_rate,
        )(inputs=x.clone(), labels=y)

        # torchattack's implementation can return nans
        ta_adv = (
            torch.where(ta_adv.isnan(), x, ta_adv),
            time.time() - start,
            "Torchattacks",
        )
    return tuple(
        [fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None]
        + [aml_adv]
    )


def df(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with DF (DeepFool)
    (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
    include ART, Foolbox, and Torchattacks. Notably, the DF implementation in
    ART, Foolbox, and Torchattacks have an overshoot parameter which we set to
    aml DF's learning rate alpha minus one (not to be confused with the epsilon
    parameter used in aml, which governs the norm-ball size).

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: DeepFool adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    df = aml.attacks.df(**parameters)
    model, max_iter, epsilon, nb_grads = (
        df.model,
        df.epochs,
        df.alpha - 1,
        df.model.classes,
    )
    art_adv = fb_adv = ta_adv = None
    start = time.time()
    aml_adv = (x + df.craft(x, y), time.time() - start, "aml")
    if "art" in frameworks:
        from art.attacks.evasion import DeepFool

        print("Producing DF adversarial examples with ART...", end=end)
        start = time.time()
        art_adv = (
            torch.from_numpy(
                DeepFool(
                    classifier=art_classifier,
                    max_iter=max_iter,
                    epsilon=epsilon,
                    nb_grads=nb_grads,
                    batch_size=x.size(0),
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import L2DeepFoolAttack as DeepFoolAttack

        print("Producing DF adversarial examples with Foolbox...", end=end)
        start = time.time()
        _, fb_adv, _ = DeepFoolAttack(
            steps=max_iter,
            candidates=nb_grads,
            overshoot=epsilon,
            loss="logits",
        )(fb_classifier, x.clone(), y.clone(), epsilons=df.epsilon)
        fb_adv = (fb_adv, time.time() - start, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import DeepFool

        print("Producing DF adversarial examples with Torchattacks...", end=end)
        start = time.time()
        ta_adv = DeepFool(
            model=model,
            steps=max_iter,
            overshoot=epsilon,
        )(inputs=x.clone(), labels=y)

        # torchattack's implementation can return nans
        ta_adv = (
            torch.where(ta_adv.isnan(), x, ta_adv),
            time.time() - start,
            "Torchattacks",
        )
    return tuple([fw for fw in (art_adv, fb_adv, ta_adv) if fw is not None] + [aml_adv])


def fab(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with FAB (Fast Adaptive Boundary)
    (https://arxiv.org/pdf/1907.02044.pdf). The supported frameworks for FAB
    include AdverTorch and Torchattacks.

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: FAB adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    fab = aml.attacks.fab(**parameters)
    (model, n_restarts, n_iter, eps, alpha, eta, beta, n_classes) = (
        fab.model,
        fab.num_restarts,
        fab.epochs,
        fab.epsilon,
        fab.traveler.optimizer.param_groups[0]["alpha_max"],
        fab.alpha,
        fab.traveler.optimizer.param_groups[0]["beta"],
        fab.model.classes,
    )
    at_adv = ta_adv = None
    reset_seeds()
    start = time.time()
    aml_adv = (x + fab.craft(x, y), time.time() - start, "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import FABAttack

        print("Producing FAB adversarial examples with AdverTorch...", end=end)
        reset_seeds()
        start = time.time()
        at_adv = (
            FABAttack(
                predict=model,
                norm="L2",
                n_restarts=n_restarts,
                n_iter=n_iter,
                eps=eps,
                alpha_max=alpha,
                eta=eta,
                beta=beta,
                verbose=verbose,
            ).perturb(x=x.clone(), y=y.clone()),
            time.time() - start,
            "AdverTorch",
        )
    if "torchattacks" in frameworks:
        from torchattacks import FAB

        print("Producing FAB adversarial examples with Torchattacks...", end=end)
        reset_seeds()
        start = time.time()
        ta_adv = (
            FAB(
                model=model,
                norm="L2",
                eps=eps,
                steps=n_iter,
                n_restarts=n_restarts,
                alpha_max=alpha,
                eta=eta,
                beta=beta,
                verbose=verbose,
                seed=0,
                multi_targeted=False,
                n_classes=n_classes,
            )(inputs=x.clone(), labels=y),
            time.time() - start,
            "Torchattacks",
        )
    reset_seeds()
    return tuple([fw for fw in (at_adv, ta_adv) if fw is not None] + [aml_adv])


def init_art_classifier(clip, device, model, features):
    """
    This method instantiates an art (pytorch) classifier (as required by art
    evasion attacks).

    :param clip: minimum and maximum feature ranges
    :type clip: torch Tensor object (2, m)
    :param device: hardware device to use
    :param device: str
    :param features: number of features (used for determing budgets)
    :type features: int
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :return: an art (PyTorch) classifier
    :rtype: art.estimator.classification PyTorchClassifier object
    """
    from art.estimators.classification import PyTorchClassifier

    mins, maxs = clip.unbind()
    return PyTorchClassifier(
        model=model.model,
        clip_values=(mins.max().item(), maxs.min().item()),
        loss=model.loss,
        optimizer=model.optimizer,
        input_shape=(features,),
        nb_classes=model.classes,
        device_type="gpu" if device == "cuda" else device,
    )


def init_attacks(
    alpha, budget, clip, device, epochs, features, frameworks, model, verbose
):
    """
    This function instantiates attacks from aml and other supported libraries.
    Specifically, it: (1) computes lp budgets and (2) prepares framework
    prerequisites.

    :param alpha: perturbation strength per-iteration
    :type alpha: float
    :param budget: maximum lp budget
    :type budget: float
    :param clip: minimum and maximum feature ranges
    :type clip: torch Tensor object (2, m)
    :param device: hardware device to use
    :param device: str
    :param epochs: number of attack iterations
    :type epochs: int
    :param dataset: dataset to use
    :type dataset: str
    :param features: number of features (used for determing budgets)
    :type features: int
    :param frameworks: frameworks to compare against
    :type frameworks: list of str
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :return: instantiated attacks
    :rtype: tuple of:
        - dict
        - art.estimators.classification PyTorchClassifier object
        - foolbox PyTorchModel object
        - int
        - float
        - float
    """
    l0 = int(features * budget) + 1
    l2 = clip.diff(dim=0).norm(2).item() * budget
    linf = budget
    params = {
        "alpha": alpha,
        "epochs": epochs,
        "model": model,
        "verbosity": float(verbose),
    }
    art_model = (
        init_art_classifier(clip, device, model, features)
        if "art" in frameworks
        else None
    )
    fb_model = (
        init_fb_classifier(clip, device, model) if "foolbox" in frameworks else None
    )
    return params, art_model, fb_model, l0, l2, linf


def init_data(dataset, device, pretrained, utilization, verbose):
    """
    This function obtains all necessary prerequisites for crafting adversarial
    examples. Specifically, this: (1) loads data, (2) determines clipping
    bounds, and (3) trains a model (if a model hasn't already been trained or
    pretrained is false).

    :param dataset: dataset to load, analyze, and train with
    :type dataset: str
    :param device: hardware device to use
    :param device: str
    :param pretrained: use a pretrained model, if possible
    :type pretrained: bool
    :param utilization: target gpu memory utilization (useful with low vram)
    :type utilization: float
    :param verbose: enable verbose output
    :type verbose: bool
    :return: test data, clips, model, and test accuracy
    :rtype: tuple of:
        - tuple of torch Tensor objects (n, m) and (n,)
        - torch Tensor object (2, m)
        - dlm LinearClassifier-inherited object
        - float
    """

    # load dataset and determine clips (non-image datasets may not be 0-1)
    data = getattr(mlds, dataset)
    try:
        train_x = torch.from_numpy(data.train.data).to(device)
        train_y = torch.from_numpy(data.train.labels).long().to(device)
        x = torch.from_numpy(data.test.data).to(device)
        y = torch.from_numpy(data.test.labels).long().to(device)
    except AttributeError:
        train_x = torch.from_numpy(data.dataset.data).to(device)
        train_y = torch.from_numpy(data.dataset.labels).long().to(device)
        x = train_x.clone().to(device)
        y = train_y.clone().to(device)
    clip = torch.stack((x.min(0).values.clamp(max=0), x.max(0).values.clamp(min=1)))

    # load model hyperparameters and train a model (or load a saved one)
    params = dict(
        auto_batch=utilization, device=device, verbosity=0.25 if verbose else 0
    )
    template = getattr(dlm.templates, dataset)
    model = (
        dlm.CNNClassifier(**template.cnn | params)
        if hasattr(template, "cnn")
        else dlm.MLPClassifier(**template.mlp | params)
    )
    try:
        if pretrained:
            with open(f"/tmp/framework_comparison_{dataset}_model.pkl", "rb") as f:
                model = pickle.load(f)
                model.summary()
        else:
            raise FileNotFoundError("Pretrained flag was false")
    except FileNotFoundError:
        model.fit(train_x, train_y)
    with open(f"/tmp/framework_comparison_{dataset}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    test_acc = model.accuracy(x, y) * 100
    return (x, y), clip, model, test_acc.item()


def init_fb_classifier(clip, device, model):
    """
    This method instantiates an art (pytorch) classifier (as required by foolbox
    evasion attacks).

    :param clip: minimum and maximum feature ranges
    :type clip: torch Tensor object (2, m)
    :param device: hardware device to use
    :param device: str
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :return: a foolbox (PyTorch) classifier
    :rtype: foolbox PyTorchModel object
    """
    from foolbox import PyTorchModel

    mins, maxs = clip
    return PyTorchModel(
        model=model.model,
        bounds=(mins.min().item(), maxs.max().item()),
        device=device,
    )


def jsma(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with JSMA (Jacobian Saliency Map
    Approach) (https://arxiv.org/pdf/1511.07528.pdf). The supported frameworks
    for the JSMA include AdverTorch and ART. Notably, the JSMA implementation
    in AdverTorch and ART both assume the l0-norm is passed in as a percentage
    and we set theta to be 1 since features can only be perturbed once.
    moreover, the advertorch implementation: (1) does not suport an untargetted
    scheme (so we supply random targets), and (2) computes every pixel pair at
    once, often leading to memory crashes (e.g., it'll terminate on a 16GB
    system with MNIST).


    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: JSMA adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    clip_min, clip_max = clip.unbind()
    jsma = aml.attacks.jsma(**parameters | dict(alpha=1))
    (model, num_classes, gamma, theta) = (
        jsma.model,
        jsma.model.classes,
        jsma.epsilon / x.size(1),
        jsma.alpha,
    )
    at_adv = art_adv = None
    start = time.time()
    aml_adv = (x + jsma.craft(x, y), time.time() - start, "aml")
    if "advertorch" in frameworks and x.size(1) < 784:
        from advertorch.attacks import JacobianSaliencyMapAttack

        print("Producing JSMA adversarial examples with AdverTorch...", end=end)
        start = time.time()
        at_adv = (
            JacobianSaliencyMapAttack(
                predict=model,
                num_classes=num_classes,
                clip_min=clip_min,
                clip_max=clip_max,
                gamma=gamma,
                theta=theta,
            ).perturb(x=x.clone(), y=torch.randint(num_classes, y.size())),
            time.time() - start,
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import SaliencyMapMethod

        print("Producing JSMA adversarial examples with ART...", end=end)
        start = time.time()
        art_adv = (
            torch.from_numpy(
                SaliencyMapMethod(
                    classifier=art_classifier,
                    theta=theta,
                    gamma=gamma,
                    batch_size=x.size(0),
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    return tuple([fw for fw in (at_adv, art_adv) if fw is not None] + [aml_adv])


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
    pretrained,
    trials,
    utilization,
    verbose,
):
    """
    This function is the main entry point for the framework comparison
    benchmark. Specifically this: (1) loads and trains a model, (2) crafts
    adversarial examples for each framework, (3) measures statistics and
    assembles dataframes, and (4) plots the results.

    :param alpha: perturbation strength, per-iteration
    :type alpha: float
    :param attacks: attacks to use
    :type attacks: list of str
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
    :param pretrained: use a pretrained model, if possible
    :type pretrained: bool
    :param trials: number of experiment trials
    :type trials: int
    :param utilization: target gpu memory utilization (useful with low vram)
    :type utilization: float
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :return: None
    :rtype: NoneType
    """
    print(
        f"Using {len(attacks)} attacks on {dataset} with "
        f"{len(frameworks)} frameworks in {trials} trials..."
    )

    # create results dataframe & setup budget measures
    metrics = "attack", "baseline", "framework", "accuracy", "budget", "time"
    results = pandas.DataFrame(columns=metrics)
    norms = collections.namedtuple("norms", ("budget", "projection", "ord"))
    norms = {
        apgdce: norms(None, linf_proj, torch.inf),
        apgddlr: norms(None, linf_proj, torch.inf),
        bim: norms(None, linf_proj, torch.inf),
        cwl2: norms(None, l2_proj, 2),
        df: norms(None, l2_proj, 2),
        fab: norms(None, l2_proj, 2),
        jsma: norms(None, l0_proj, 0),
        pgd: norms(None, linf_proj, torch.inf),
    }
    end = "\n" if verbose else "\r"
    row = 0

    # load data, clipping bounds, attacks, and train a model
    for t in range(trials):
        print(f"Preparing {dataset} model... Trial {t} of {trials}", end=end)
        (x, y), clip, model, test_acc = init_data(
            dataset, device, pretrained, utilization, verbose
        )
        parameters, art_model, fb_model, l0, l2, linf = init_attacks(
            alpha,
            budget,
            clip,
            device,
            epochs,
            x.size(1),
            frameworks,
            model,
            verbose,
        )
        for a, n in norms.items():
            norms[a] = n._replace(
                budget=linf if n.ord == torch.inf else l2 if n.ord == 2 else l0
            )

        # craft adversarial examples
        for j, a in enumerate(attacks):
            print(f"Attacking with {a.__name__}... {j} of {len(attacks)}", end=end)
            advs = a(
                art_model,
                clip,
                fb_model,
                frameworks,
                parameters | dict(epsilon=norms[a].budget),
                verbose,
                x,
                y,
            )

            # ensure adversarial examples comply with clips and epsilon
            for adv, times, fw in advs:
                print(f"Computing results for {fw} {a.__name__}...", end=end)
                adv = adv.to(device).clamp(*clip.unbind())
                p = norms[a].projection(norms[a].budget, adv.sub(x))
                acc = model.accuracy(x + p, y).mul(100).item()
                used = (
                    p.norm(norms[a].ord, 1).mean().div(norms[a].budget).mul(100).item()
                )
                results.loc[row] = a.__name__, test_acc, fw, acc, used, times
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


def pgd(art_classifier, clip, fb_classifier, frameworks, parameters, verbose, x, y):
    """
    This method crafts adversarial examples with PGD (Projected Gradient
    Descent)) (https://arxiv.org/pdf/1706.06083.pdf). The supported frameworks
    for PGD include AdverTorch, ART, CleverHans, and Torchattacks.

    :param art_classifier: classifier for ART
    :type art_classifier: art.estimator.classification PyTorchClassifier object
    :param clip: allowable feature range for the domain
    :type clip: torch Tensor object (2, m)
    :param fb_classifier: classifier for Foolbox
    :type fb_classifier: foolbox PyTorchModel object
    :param frameworks: frameworks to craft adversarial examples with
    :type frameworks: tuple of str
    :param parameters: attack parameters
    :type parameters: dict
    :param verbose: enable verbose output
    :tyhpe verbose: bool
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: PGD adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m), float, and str
    """
    end = "\n" if verbose else "\r"
    clip_min, clip_max = clip.unbind()
    pgd = aml.attacks.pgd(**parameters)
    (model, eps, nb_iter, eps_iter) = (
        pgd.model,
        pgd.epsilon,
        pgd.epochs,
        pgd.alpha,
    )
    at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
    reset_seeds()
    start = time.time()
    aml_adv = (x + pgd.craft(x, y), time.time() - start, "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import PGDAttack

        print("Producing PGD adversarial examples with AdverTorch...", end=end)
        reset_seeds()
        start = time.time()
        at_adv = (
            PGDAttack(
                predict=model,
                loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                eps=eps,
                nb_iter=nb_iter,
                eps_iter=eps_iter,
                rand_init=True,
                clip_min=clip_min,
                clip_max=clip_max,
                ord=float("inf"),
                targeted=False,
            ).perturb(x=x.clone(), y=y.clone()),
            time.time() - start,
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import ProjectedGradientDescent

        print("Producing PGD adversarial examples with ART...", end=end)
        reset_seeds()
        start = time.time()
        art_adv = (
            torch.from_numpy(
                ProjectedGradientDescent(
                    estimator=art_classifier,
                    norm="inf",
                    eps=eps,
                    eps_step=eps_iter,
                    decay=None,
                    max_iter=nb_iter,
                    targeted=False,
                    num_random_init=1,
                    batch_size=x.size(0),
                    random_eps=True,
                    summary_writer=False,
                    verbose=verbose,
                ).generate(x=x.clone().cpu().numpy())
            ),
            time.time() - start,
            "ART",
        )
    if "cleverhans" in frameworks:
        from cleverhans.torch.attacks.projected_gradient_descent import (
            projected_gradient_descent,
        )

        print("Producing PGD adversarial examples with CleverHans...", end=end)
        reset_seeds()
        start = time.time()
        ch_adv = (
            projected_gradient_descent(
                model_fn=model,
                x=x.clone(),
                eps=eps,
                eps_iter=eps_iter,
                nb_iter=nb_iter,
                norm=float("inf"),
                clip_min=clip_min.max(),
                clip_max=clip_max.min(),
                y=y,
                targeted=False,
                rand_init=True,
                rand_minmax=eps,
                sanity_checks=False,
            ).detach(),
            time.time() - start,
            "CleverHans",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import (
            LinfProjectedGradientDescentAttack as ProjectedGradientDescentAttack,
        )

        print("Producing PGD adversarial examples with Foolbox...", end=end)
        reset_seeds()
        start = time.time()
        _, fb_adv, _ = ProjectedGradientDescentAttack(
            rel_stepsize=None,
            abs_stepsize=eps_iter,
            steps=nb_iter,
            random_start=True,
        )(
            fb_classifier,
            x.clone(),
            y.clone(),
            epsilons=eps,
        )
        fb_adv = (fb_adv, time.time() - start, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import PGD

        print("Producing PGD adversarial examples with Torchattacks...", end=end)
        reset_seeds()
        start = time.time()
        ta_adv = (
            PGD(
                model=model,
                eps=eps,
                alpha=eps_iter,
                steps=nb_iter,
                random_start=True,
            )(inputs=x.clone(), labels=y),
            time.time() - start,
            "Torchattacks",
        )
    reset_seeds()
    return tuple(
        [aml_adv]
        + [fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None]
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
    frameworks = ("advertorch", "art", "cleverhans", "foolbox", "torchattacks")
    available_frameworks = [f for f in frameworks if importlib.util.find_spec(f)]

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
        choices=available_frameworks,
        default=available_frameworks,
        help="Frameworks to compare against",
        nargs="+",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        action="store_true",
        help="Avoid training a new model, if possible (helpful for debugging)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        help="Number of experiment trials (set to 1 if pretrained is true)",
        type=int,
    )
    parser.add_argument(
        "-u",
        "--utilization",
        default=1.0,
        help="Target GPU utilization (useful with GPUs that have low VRAM)",
        type=float,
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
        pretrained=args.pretrained,
        trials=1 if args.pretrained else args.trials,
        utilization=args.utilization,
        verbose=args.verbose,
    )
    raise SystemExit(0)
