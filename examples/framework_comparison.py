"""
This script compares the performance of the aml framework against other popular
frameworks and plots the model accuracy over the norm.
Author: Ryan Sheatsley
Sat Feb 4 2023
"""
import argparse
import collections
import importlib
import pickle
import warnings

import aml
import dlm
import matplotlib.ticker
import mlds
import numpy as np
import pandas
import seaborn
import torch

# dlm uses lazy modules which induce warnings that overload stdout
warnings.filterwarnings("ignore", category=UserWarning)


def apgdce(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: APGD-CE adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
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
    aml_adv = (x + apgdce.craft(x, y), "aml")
    if "art" in frameworks:
        from art.attacks.evasion import AutoProjectedGradientDescent

        print("Producing APGD-CE adversarial examples with ART...", end="\r")
        reset_seeds()
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
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
            "ART",
        )
    if "torchattacks" in frameworks and hasattr(model, "shape"):
        from torchattacks import APGD

        print("Producing APGD-CE adversarial examples with Torchattacks...", end="\r")
        reset_seeds()
        ta_x = x.clone().unflatten(1, model.shape)
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
                verbose=False,
            )(inputs=ta_x, labels=y)
            .flatten(1)
            .detach(),
            "Torchattacks",
        )
    reset_seeds()
    return tuple([fw for fw in (art_adv, ta_adv) if fw is not None] + [aml_adv])


def apgddlr(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: APGD-DLR adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
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
    aml_adv = (x + apgddlr.craft(x, y), "aml")
    if "art" in frameworks and art_classifier.nb_classes > 2:
        from art.attacks.evasion import AutoProjectedGradientDescent

        print("Producing APGD-DLR adversarial examples with ART...", end="\r")
        reset_seeds()
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
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
            "ART",
        )
    if (
        "torchattacks" in frameworks
        and art_classifier.nb_classes > 2
        and hasattr(model, "shape")
    ):
        from torchattacks import APGD

        print("Producing APGD-DLR adversarial examples with Torchattacks...", end="\r")
        reset_seeds()
        ta_x = x.clone().unflatten(1, model.shape)
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
                verbose=False,
            )(inputs=ta_x, labels=y)
            .flatten(1)
            .detach(),
            "Torchattacks",
        )
    reset_seeds()
    return tuple([fw for fw in (art_adv, ta_adv) if fw is not None] + [aml_adv])


def bim(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: BIM adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
    clip_min, clip_max = clip.unbind()
    bim = aml.attacks.bim(**parameters)
    model, eps, nb_iter, eps_iter = bim.model, bim.epsilon, bim.epochs, bim.alpha
    at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
    aml_adv = (x + bim.craft(x, y), "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import LinfBasicIterativeAttack as BasicIterativeAttack

        print("Producing BIM adversarial examples with AdverTorch...", end="\r")
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
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import BasicIterativeMethod

        print("Producing BIM adversarial examples with ART...", end="\r")
        art_adv = (
            torch.from_numpy(
                BasicIterativeMethod(
                    estimator=art_classifier,
                    eps=eps,
                    eps_step=eps_iter,
                    max_iter=nb_iter,
                    targeted=False,
                    batch_size=x.size(0),
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
            "ART",
        )
    if "cleverhans" in frameworks:
        from cleverhans.torch.attacks.projected_gradient_descent import (
            projected_gradient_descent as basic_iterative_method,
        )

        print("Producing BIM adversarial examples with CleverHans...", end="\r")
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
            "CleverHans",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import LinfBasicIterativeAttack as BasicIterativeAttack

        print("Producing BIM adversarial examples with Foolbox...", end="\r")
        _, fb_adv, _ = BasicIterativeAttack(
            rel_stepsize=None,
            abs_stepsize=eps_iter,
            steps=nb_iter,
            random_start=False,
        )(fb_classifier, x.clone(), y.clone(), epsilons=eps)
        fb_adv = (fb_adv, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import BIM

        print("Producing BIM adversarial examples with Torchattacks...", end="\r")
        ta_adv = (
            BIM(model=model, eps=eps, alpha=eps_iter, steps=nb_iter)(
                inputs=x.clone(), labels=y
            ),
            "Torchattacks",
        )
    return tuple(
        [fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None]
        + [aml_adv]
    )


def cwl2(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: CW-L2 adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
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
    aml_adv = (x + cwl2.craft(x, y), "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import CarliniWagnerL2Attack

        print("Producing CW-L2 adversarial examples with AdverTorch...", end="\r")
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
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import CarliniL2Method as CarliniWagner

        print("Producing CW-L2 adversarial examples with ART...", end="\r")
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
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
            "ART",
        )
    if "cleverhans" in frameworks:
        from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

        print("Producing CW-L2 adversarial examples with CleverHans...", end="\r")
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
            "CleverHans",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import L2CarliniWagnerAttack

        print("Producing CW-L2 adversarial examples with Foolbox...", end="\r")
        _, fb_adv, _ = L2CarliniWagnerAttack(
            binary_search_steps=binary_search_steps,
            steps=max_iterations,
            stepsize=learning_rate,
            confidence=confidence,
            initial_const=initial_const,
            abort_early=True,
        )(fb_classifier, x.clone(), y.clone(), epsilons=cwl2.epsilon)
        fb_adv = (fb_adv, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import CW

        print("Producing CW-L2 adversarial examples with Torchattacks...", end="\r")
        ta_adv = CW(
            model=model,
            c=initial_const,
            kappa=confidence,
            steps=max_iterations,
            lr=learning_rate,
        )(inputs=x.clone(), labels=y)

        # torchattack's implementation can return nans
        ta_adv = (torch.where(ta_adv.isnan(), x, ta_adv), "Torchattacks")
    return tuple(
        [fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None]
        + [aml_adv]
    )


def df(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: DeepFool adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
    df = aml.attacks.df(**parameters)
    model, max_iter, epsilon, nb_grads = (
        df.model,
        df.epochs,
        df.alpha - 1,
        df.model.classes,
    )
    art_adv = fb_adv = ta_adv = None
    aml_adv = (x + df.craft(x, y), "aml")
    if "art" in frameworks:
        from art.attacks.evasion import DeepFool

        print("Producing DF adversarial examples with ART...", end="\r")
        art_adv = (
            torch.from_numpy(
                DeepFool(
                    classifier=art_classifier,
                    max_iter=max_iter,
                    epsilon=epsilon,
                    nb_grads=nb_grads,
                    batch_size=x.size(0),
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
            "ART",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import L2DeepFoolAttack as DeepFoolAttack

        print("Producing DF adversarial examples with Foolbox...", end="\r")
        _, fb_adv, _ = DeepFoolAttack(
            steps=max_iter,
            candidates=nb_grads,
            overshoot=epsilon,
            loss="logits",
        )(fb_classifier, x.clone(), y.clone(), epsilons=df.epsilon)
        fb_adv = (fb_adv, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import DeepFool

        print("Producing DF adversarial examples with Torchattacks...", end="\r")
        ta_adv = DeepFool(
            model=model,
            steps=max_iter,
            overshoot=epsilon,
        )(inputs=x.clone(), labels=y)

        # torchattack's implementation can return nans
        ta_adv = (torch.where(ta_adv.isnan(), x, ta_adv), "Torchattacks")
    return tuple([fw for fw in (art_adv, fb_adv, ta_adv) if fw is not None] + [aml_adv])


def fab(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: FAB adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
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
    aml_adv = (x + fab.craft(x, y), "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import FABAttack

        print("Producing FAB adversarial examples with AdverTorch...", end="\r")
        reset_seeds()
        at_adv = (
            FABAttack(
                predict=model,
                norm="Linf",
                n_restarts=n_restarts,
                n_iter=n_iter,
                eps=eps,
                alpha_max=alpha,
                eta=eta,
                beta=beta,
                verbose=False,
            ).perturb(x=x.clone(), y=y.clone()),
            "AdverTorch",
        )
    if "torchattacks" in frameworks:
        from torchattacks import FAB

        print("Producing FAB adversarial examples with Torchattacks...", end="\r")
        reset_seeds()
        ta_adv = (
            FAB(
                model=model,
                norm="Linf",
                eps=eps,
                steps=n_iter,
                n_restarts=n_restarts,
                alpha_max=alpha,
                eta=eta,
                beta=beta,
                verbose=False,
                seed=0,
                multi_targeted=False,
                n_classes=n_classes,
            )(inputs=x.clone(), labels=y),
            "Torchattacks",
        )
    reset_seeds()
    return tuple([fw for fw in (at_adv, ta_adv) if fw is not None] + [aml_adv])


def init_art_classifier(clip, model, features):
    """
    This method instantiates an art (pytorch) classifier (as required by art
    evasion attacks).

    :param clip: minimum and maximum feature ranges
    :type clip: torch Tensor object (2, m)
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
    )


def init_attacks(alpha, budget, clip, epochs, features, frameworks, model):
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
    params = {"alpha": alpha, "epochs": epochs, "model": model}
    art_model = (
        init_art_classifier(clip, model, features) if "art" in frameworks else None
    )
    fb_model = init_fb_classifier(clip, model) if "foolbox" in frameworks else None
    return params, art_model, fb_model, l0, l2, linf


def init_data(dataset, pretrained):
    """
    This function obtains all necessary prerequisites for crafting adversarial
    examples. Specifically, this: (1) loads data, (2) determines clipping
    bounds, and (3) trains a model (if a model hasn't already been trained or
    pretrained is false).

    :param dataset: dataset to load, analyze, and train with
    :type dataset: str
    :param pretrained: use a pretrained model, if possible
    :type pretrained: bool
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
        train_x = torch.from_numpy(data.train.data)
        train_y = torch.from_numpy(data.train.labels).long()
        x = torch.from_numpy(data.test.data)
        y = torch.from_numpy(data.test.labels).long()
    except AttributeError:
        train_x = torch.from_numpy(data.dataset.data)
        train_y = torch.from_numpy(data.dataset.labels).long()
        x = train_x.clone()
        y = train_y.clone()
    clip = torch.stack((x.min(0).values.clamp(max=0), x.max(0).values.clamp(min=1)))

    # load model hyperparameters and train a model (or load a saved one)
    template = getattr(dlm.templates, dataset)
    model = (
        dlm.CNNClassifier(**template.cnn)
        if hasattr(template, "cnn")
        else dlm.MLPClassifier(**template.mlp)
    )
    model.verbosity = 0
    try:
        if pretrained:
            with open(f"/tmp/framework_comparison_{dataset}_model.pkl", "rb") as f:
                model = pickle.load(f)
                model.summary()
        else:
            raise FileNotFoundError("Pretrained flag was false")
    except FileNotFoundError as e:
        print(f"Training a new model ({e})")
        model.fit(train_x, train_y)
    with open(f"/tmp/framework_comparison_{dataset}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    test_acc = model.accuracy(x, y)
    return (x, y), clip, model, test_acc.item()


def init_fb_classifier(clip, model):
    """
    This method instantiates an art (pytorch) classifier (as required by foolbox
    evasion attacks).

    :param clip: minimum and maximum feature ranges
    :type clip: torch Tensor object (2, m)
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
    )


def jsma(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: JSMA adversarial examples
    :rtype: tuple of tuples: torch Tensor object (n, m) and str
    """
    clip_min, clip_max = clip.unbind()
    jsma = aml.attacks.jsma(**parameters | dict(alpha=1))
    (model, num_classes, gamma, theta) = (
        jsma.model,
        jsma.model.classes,
        jsma.epsilon / x.size(1),
        jsma.alpha,
    )
    at_adv = art_adv = None
    aml_adv = (x + jsma.craft(x, y), "aml")
    if "advertorch" in frameworks and x.size(1) < 784:
        from advertorch.attacks import JacobianSaliencyMapAttack

        print("Producing JSMA adversarial examples with AdverTorch...", end="\r")
        at_adv = (
            JacobianSaliencyMapAttack(
                predict=model,
                num_classes=num_classes,
                clip_min=clip_min,
                clip_max=clip_max,
                gamma=gamma,
                theta=theta,
            ).perturb(x=x.clone(), y=torch.randint(num_classes, y.size())),
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import SaliencyMapMethod

        print("Producing JSMA adversarial examples with ART...", end="\r")
        art_adv = (
            torch.from_numpy(
                SaliencyMapMethod(
                    classifier=art_classifier,
                    theta=theta,
                    gamma=gamma,
                    batch_size=x.size(0),
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
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


def main(alpha, attacks, budget, datasets, epochs, frameworks, pretrained, trials):
    """
    This function is the main entry point for the framework comparison
    benchmark. Specifically this: (1) loads and trains a model for each
    dataset, (2) crafts adversarial examples for each framework, (3) measures
    statistics and assembles dataframes, and (4) plots the results.

    :param alpha: perturbation strength, per-iteration
    :type alpha: float
    :param attacks: attacks to use
    :type attacks: list of str
    :param budget: maximum lp budget
    :type budget: float
    :param datasets: dataset(s) to use
    :type datasets: tuple of str
    :param epochs: number of attack iterations
    :type epochs: int
    :param frameworks: frameworks to compare against
    :type frameworks: tuple of str
    :param pretrained: use a pretrained model, if possible
    :type pretrained: bool
    :param trials: number of experiment trials
    :type trials: int
    :return: None
    :rtype: NoneType
    """
    print(
        f"Analyzing {len(attacks)} attacks across {len(datasets)} datasets with "
        f"{len(frameworks)} frameworks in {trials} trials..."
    )
    metrics = "dataset", "attack", "baseline", "framework", "accuracy", "budget"
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
    row = 0
    for d in datasets:

        # load data, clipping bounds, attacks, and train a model
        for t in range(trials):
            print(f"Preparing {d} model... Trial {t} of {trials}", end="\r")
            (x, y), clip, model, test_acc = init_data(d, pretrained)
            parameters, art_model, fb_model, l0, l2, linf = init_attacks(
                alpha, budget, clip, epochs, x.size(1), frameworks, model
            )
            for a, n in norms.items():
                norms[a] = n._replace(
                    budget=linf if n.ord == torch.inf else l2 if n.ord == 2 else l0
                )

            # craft adversarial examples
            for j, a in enumerate(attacks):
                print(f"Attacking with {a.__name__}... {j} of {len(attacks)}", end="\r")
                advs = a(
                    art_model,
                    clip,
                    fb_model,
                    frameworks,
                    parameters | dict(epsilon=norms[a].budget),
                    x,
                    y,
                )

                # ensure adversarial examples comply with clips and epsilon
                for adv, fw in advs:
                    print(f"Computing results for {fw} {a.__name__}...", end="\r")
                    adv = adv.clamp(*clip.unbind())
                    p = norms[a].projection(norms[a].budget, adv.sub(x))
                    acc = model.accuracy(x + p, y).item()
                    used = p.norm(norms[a].ord, 1).mean().div(norms[a].budget).item()
                    results.loc[row] = d, a.__name__, test_acc, fw, acc, used
                    row += 1

    # take the median of the results, plot, and save
    results = results.groupby(["dataset", "attack", "framework"]).median().reset_index()
    plot(results)
    return None


def plot(results):
    """
    This function plots the framework comparison results. Specifically, this
    produces  one scatter plot per dataset containing model accuracy on the
    crafting set over the percentage of the budget consumed (with a dotted line
    designating the original model accuracy on the clean data). Frameworks
    labeled above points and are divided by color, while attacks are divided by
    marker style. Axes are in log scale to show separation. The plot is written
    to disk in the current directory.

    :param results: results of the framework comparison
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    plot = seaborn.relplot(
        data=results,
        col="dataset",
        col_wrap=(results.dataset.unique().size + 1) // 2,
        facet_kws=dict(subplot_kws=dict(xscale="log", yscale="log")),
        hue="framework",
        kind="scatter",
        legend="full" if results.dataset.unique().size > 1 else "auto",
        style="attack",
        x="budget",
        y="accuracy",
        **dict(alpha=0.7, s=100),
    )
    plot.map_dataframe(
        color="r",
        func=seaborn.lineplot,
        linestyle="--",
        x="budget",
        y="baseline",
    )
    plot.legend.remove()
    plot._legend_data["baseline"] = plot._legend_data["framework"]
    plot._legend_data["clean"] = plot.axes[0].lines[0]
    plot.add_legend(adjust_subtitles=True)
    plot.set(ylabel="accuracy")
    for ax in plot.axes.flat:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(1, 3, 5, 8)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    plot.savefig(__file__[:-2] + ".pdf", bbox_inches="tight")
    return None


def pgd(art_classifier, clip, fb_classifier, frameworks, parameters, x, y):
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
    :param x: inputs to craft adversarial examples from
    :type x: torch Tensor object (n, m)
    :param y: labels of inputs to craft adversarail examples from
    :type x: torch Tensor object (n,)
    :return: PGD adversarial examples
    :rtype: tuple of tuples: aml Attack object & torch Tensor object (n, m)
    """
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
    aml_adv = (x + pgd.craft(x, y), "aml")
    if "advertorch" in frameworks:
        from advertorch.attacks import PGDAttack

        print("Producing PGD adversarial examples with AdverTorch...", end="\r")
        reset_seeds()
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
            "AdverTorch",
        )
    if "art" in frameworks:
        from art.attacks.evasion import ProjectedGradientDescent

        print("Producing PGD adversarial examples with ART...", end="\r")
        reset_seeds()
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
                    verbose=False,
                ).generate(x=x.clone().numpy())
            ),
            "ART",
        )
    if "cleverhans" in frameworks:
        from cleverhans.torch.attacks.projected_gradient_descent import (
            projected_gradient_descent,
        )

        print("Producing PGD adversarial examples with CleverHans...", end="\r")
        reset_seeds()
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
            "CleverHans",
        )
    if "foolbox" in frameworks:
        from foolbox.attacks import (
            LinfProjectedGradientDescentAttack as ProjectedGradientDescentAttack,
        )

        print("Producing PGD adversarial examples with Foolbox...", end="\r")
        reset_seeds()
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
        fb_adv = (fb_adv, "Foolbox")
    if "torchattacks" in frameworks:
        from torchattacks import PGD

        print("Producing PGD adversarial examples with Torchattacks...", end="\r")
        reset_seeds()
        ta_adv = (
            PGD(
                model=model,
                eps=eps,
                alpha=eps_iter,
                steps=nb_iter,
                random_start=True,
            )(inputs=x.clone(), labels=y),
            "Torchattacks",
        )
    reset_seeds()
    return tuple(
        [aml_adv]
        + [fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None]
    )


def reset_seeds():
    """
    This method resets the seeds for random number generators used in attacks
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
    this script: (1) parses command-line arguments, (2) loads dataset(s), (3)
    trains a model, (4) crafts adversarial examples for each framework, (5)
    collects statistics on model accuracy and lp-norm of the adversarial
    examples, and (6) plots the results.
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
        "--datasets",
        choices=mlds.__available__,
        default=mlds.__available__,
        help="Dataset(s) to use",
        nargs="+",
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
        default=False,
        help="Avoid training a new model, if possible (helpful for debugging)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        help="Number of experiment trials (set to 1 if pretrained is true)",
        type=int,
    )
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        attacks=args.attacks,
        budget=args.budget,
        datasets=args.datasets,
        epochs=args.epochs,
        frameworks=tuple(args.frameworks),
        pretrained=args.pretrained,
        trials=1 if args.pretrained else args.trials,
    )
    raise SystemExit(0)
