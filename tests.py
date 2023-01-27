"""
This module runs attack performance and correctness tests against other AML
libraries. Specifically, this defines three types of tests: (1) functional, (2)
semantic, and (3) identity. Details surrounding these tests can be found in the
respecetive classes: FunctionalTests, SemanticTests, and IdentityTests. Aside
from these standard tests, special tests can be found in the SpecialTests
class, which evaluate particulars of the implementation in the Space of
Adversarial Strategies (https://arxiv.org/pdf/2209.04521.pdf).
Authors: Ryan Sheatsley and Blaine Hoak
Mon Nov 28 2022
"""

import aml  # ML robustness evaluations with PyTorch
import dlm  # pytorch--based deep learning models with scikit-learn-like interfaces
import importlib  # The implementation of import
import mlds  # Scripts for downloading, preprocessing, and numpy-ifying popular machine learning datasets
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import pickle  # Python object serialization
import unittest  # Unit testing framework
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# get all functional tests passing
# get all semantic tests passing
# consider special tests for component performance
# add cli


class BaseTest(unittest.TestCase):
    """
    The following class serves as the base test for all tests cases. It
    provides: (1) common initializations required by all tests, such as
    retrieving data and training models (since there is apparently no elegant
    way to set up text fixtures *across test cases*, even though setUpModule
    sounds like it ought to fit the bill...), (2) initializations required by
    some tests (e.g., instantiating objects specific to other adversarial
    machine learning frameworks), and (3) methods to craft adversarial examples
    from aml and other supported adversarial machine learning frameworks.

    The following frameworks are supported:
        AdverTorch (https://github.com/BorealisAI/advertorch)
        ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
        CleverHans (https://github.com/cleverhans-lab/cleverhans)
        Foolbox (https://github.com/bethgelab/foolbox)
        Torchattacks (https://github.com/Harry24k/adversarial-attacks-pytorch)

    The following attacks are supported:
        APGD-CE (Auto-PGD with CE loss) (https://arxiv.org/pdf/2003.01690.pdf)
        APGD-DLR (Auto-PGD with DLR loss) (https://arxiv.org/pdf/2003.01690.pdf)
        BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
        CW-L2 (Carlini-Wagner with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf)
        DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
        FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
        JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)
        PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)

    :func:`setUpClass`: initializes the setup for all tests cases
    :func:`build_art_classifier`: instantiates an art pytorch classifier
    :func:`l0_proj`: projects inputs onto l0-based threat models
    :func:`l2_proj`: projects inputs onto l2-based threat models
    :func:`linf_proj`: projects inputs onto l∞-based threat models
    :func:`reset_seeds`: resets seeds for RNGs
    :func:`apgdce`: craft adversarial examples with APGD-CE
    :func:`apgddlr`: craft adversarial examples with APGD-DLR
    :func:`bim`: craft adversarial examples with BIM
    :func:`cwl2`: craft adversarial examples with CW-L2
    :func:`df`: craft adversarial examples with DF
    :func:`fab`: craft adversarial examples with FAB
    :func:`jsma`: craft adversarial examples with JSMA
    :func:`pgd`: craft adversarial examples with PGD
    """

    @classmethod
    def setUpClass(
        cls,
        alpha=0.01,
        dataset="phishing",
        debug=False,
        epochs=30,
        norm=0.15,
        seed=5115,
        verbose=False,
    ):
        """
        This function initializes the setup necessary for all test cases within
        this module. Specifically, this method retrieves data, processes it,
        trains a model, instantiates attacks, imports availabe external
        libraries (used for semantic and identity tests), loads PyTorch in
        debug mode if desired (to assist debugging autograd), and sets the seed
        used to make attacks with randomized components deterministic.

        :param alpha: perturbation strength
        :type alpha: float
        :param dataset: dataset to run tests over
        :type dataset: str
        :param debug: whether to set the autograd engine in debug mode
        :type debugf: bool
        :param epochs: number of attack iterations
        :type epochs: int
        :param norm: maximum % of lp-budget consumption
        :type norm: float
        :param seed: the seed to use to make randomized components determinisitc
        :type seed: int
        :param verbose: print attack status during crafting
        :type verbose float
        :return: None
        :rtype: NoneType
        """

        # set debug, seed, and load appropriate data partitions
        print("Initializing module for all test cases...")
        torch.autograd.set_detect_anomaly(debug)
        cls.seed = seed
        data = getattr(mlds, dataset)
        try:
            x = torch.from_numpy(data.train.data)
            y = torch.from_numpy(data.train.labels).long()
            cls.x = torch.from_numpy(data.test.data)
            cls.y = torch.from_numpy(data.test.labels).long()
            has_test = True
        except AttributeError:
            has_test = False
            cls.x = torch.from_numpy(data.dataset.data)
            cls.y = torch.from_numpy(data.dataset.labels).long()

        # ensure image dataset dimensions are pytorch-compliant
        shape = (
            data.orgfshape[::-1]
            if len(data.orgfshape) == 3
            else (1,) + data.orgfshape
            if len(data.orgfshape) == 2
            else None
        )

        # determine clipping range (non-image datasets may not be 0-1)
        mins = torch.zeros(cls.x.size(1)) if shape else cls.x.min(0).values.clamp(0)
        maxs = torch.ones(cls.x.size(1)) if shape else cls.x.max(0).values.clamp(1)
        clip = (mins, maxs)
        p_clip = (mins.sub(cls.x), maxs.sub(cls.x))
        clip_info = (
            (mins[0].item(), maxs[0].item())
            if (mins[0].eq(mins).all() and maxs[0].eq(maxs).all())
            else f"(({mins.min().item()}, ..., {mins.max().item()}), "
            f"({maxs.min().item()}, ..., {maxs.max().item()}))"
        )

        # load model hyperparameters
        cls.reset_seeds()
        template = getattr(dlm.architectures, dataset)
        cls.model = (
            dlm.CNNClassifier(**template.CNNClassifier)
            if template.CNNClassifier is not None
            else dlm.MLPClassifier(**template.MLPClassifier)
        )

        # train (or load) model and save craftset accuracy (for semantic tests)
        try:
            path = pathlib.Path(f"/tmp/aml_trained_{dataset}_model.pkl")
            with path.open("rb") as f:
                state = pickle.load(f)
            temp_to_apply = getattr(template, cls.model.__class__.__name__)
            assert state["template"] == temp_to_apply, "Hyperparameters have changed"
            assert state["seed"] == seed, f"Seed changed: {state['seed']} != {seed}"
            cls.model = state["model"]
            print(f"Using pretrained {dataset} model from {path}")
        except (FileNotFoundError, AssertionError) as e:
            print(f"Training new model... ({e})")
            cls.model.fit(*(x, y) if has_test else (cls.x, cls.y), shape=shape)
            state = {
                "model": cls.model,
                "template": getattr(template, cls.model.__class__.__name__),
                "seed": seed,
            }
            with path.open("wb") as f:
                pickle.dump(state, f)
        cls.clean_acc = cls.model.accuracy(cls.x, cls.y).item()

        # instantiate attacks and save attack parameters
        cls.l0_max = cls.x.size(1)
        cls.l0 = int(cls.l0_max * norm) + 1
        cls.l2_max = maxs.sub(mins).norm(2).item()
        cls.l2 = cls.l2_max * norm
        cls.linf_max = 1
        cls.linf = norm
        cls.clip_min, cls.clip_max = clip
        cls.p_min, cls.p_max = p_clip
        cls.verbose = verbose
        cls.atk_params = {
            "alpha": alpha,
            "epochs": epochs,
            "model": cls.model,
            "verbosity": 0.1 if verbose else 1,
        }

        # df & fab alpha must be >= 1 and cw must have at least 300 epochs
        cls.attacks = {
            "apgdce": aml.attacks.apgdce(**cls.atk_params | {"epsilon": cls.linf}),
            "apgddlr": aml.attacks.apgddlr(**cls.atk_params | {"epsilon": cls.linf}),
            "bim": aml.attacks.bim(**cls.atk_params | {"epsilon": cls.linf}),
            "cwl2": aml.attacks.cwl2(
                **cls.atk_params | {"epochs": max(300, epochs), "epsilon": cls.l2}
            ),
            "df": aml.attacks.df(**cls.atk_params | {"alpha": 1, "epsilon": cls.l2}),
            "fab": aml.attacks.fab(**cls.atk_params | {"alpha": 1, "epsilon": cls.l2}),
            "jsma": aml.attacks.jsma(**cls.atk_params | {"epsilon": cls.l0}),
            "pgd": aml.attacks.pgd(**cls.atk_params | {"epsilon": cls.linf}),
        }

        # determine available frameworks and set both art & foolbox classifiers
        supported = ("advertorch", "art", "cleverhans", "foolbox", "torchattacks")
        cls.available = [f for f in supported if importlib.util.find_spec(f)]
        frameworks = ", ".join(cls.available)
        cls.art_classifier = (
            cls.build_art_classifier()
            if cls in (IdentityTests, SemanticTests) and "art" in cls.available
            else None
        )
        cls.fb_classifier = (
            cls.build_fb_classifier()
            if cls in (IdentityTests, SemanticTests) and "foolbox" in cls.available
            else None
        )
        print(
            "Module Setup complete. Testing Parameters:",
            f"Dataset: {dataset}, Test Set: {has_test}",
            f"Craftset shape: ({cls.x.size(0)}, {cls.x.size(1)})",
            f"Model Type: {cls.model.__class__.__name__}",
            f"Train Acc: {cls.model.stats['train_acc'][-1]:.1%}",
            f"Craftset Acc: {cls.clean_acc:.1%}",
            f"Attack Clipping Values: {clip_info}",
            f"Attack Strength α: {cls.atk_params['alpha']}",
            f"Attack Epochs: {cls.atk_params['epochs']}",
            f"Max Norm Radii: l0: {cls.l0}, l2: {cls.l2:.3}, l∞: {cls.linf}",
            f"Available Frameworks: {frameworks}",
            sep="\n",
        )
        return None

    @classmethod
    def build_art_classifier(cls):
        """
        This method instantiates an art (pytorch) classifier (as required by art
        evasion attacks).

        :return: an art (pytorch) classifier
        :rtype: art PyTorchClassifier object
        """
        from art.estimators.classification import PyTorchClassifier

        return PyTorchClassifier(
            model=cls.atk_params["model"].model,
            clip_values=(
                min(cls.clip_min.max().item(), 0),
                max(cls.clip_max.min().item(), 1),
            ),
            loss=cls.atk_params["model"].loss,
            optimizer=cls.atk_params["model"].optimizer,
            input_shape=(cls.x.size(1),),
            nb_classes=cls.atk_params["model"].params["classes"],
        )

    @classmethod
    def build_fb_classifier(cls):
        """
        This method instantiates an art (pytorch) classifier (as required by foolbox
        evasion attacks).

        :return: an art (pytorch) classifier
        :rtype: art PyTorchClassifier object
        """
        from foolbox import PyTorchModel

        return PyTorchModel(
            model=cls.atk_params["model"].model,
            bounds=(
                min(cls.clip_min.max().item(), 0),
                max(cls.clip_max.min().item(), 1),
            ),
        )

    @classmethod
    def l0_proj(cls, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l0 threat model. Specifically, the components of
        perturbations that exceed the threat model are set to zero, sorted by
        increasing magnitude.

        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: threat-model-compliant perturbation vectors
        :rtype: torch Tensor object (n, m)
        """
        return p.scatter(1, p.abs().sort(1).indices[:, : p.size(1) - cls.l0], 0)

    @classmethod
    def l2_proj(cls, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l2 threat model Specifically,
        perturbation vectors whose l2-norms exceed the threat model are
        normalized by their l2-norms times epsilon.

        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: threat-model-compliant perturbation vectors
        :rtype: torch Tensor object (n, m)
        """
        return p.renorm(2, 0, cls.l2)

    @classmethod
    def linf_proj(cls, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l∞ threat model  Specifically,
        perturbation vectors whose l∞-norms exceed the threat model are
        are clipped to ±epsilon.

        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: threat-model-compliant perturbation vectors
        :rtype: torch Tensor object (n, m)
        """
        return p.clamp(-cls.linf, cls.linf)

    @classmethod
    def reset_seeds(cls):
        """
        This method resets the seeds for random number generators used
        in attacks with randomized components throughtout adversarial machine
        learning frameworks.

        :return: None
        :rtype: NoneType
        """
        torch.manual_seed(cls.seed)
        np.random.seed(cls.seed)

    def apgdce(self):
        """
        This method crafts adversarial examples with APGD-CE (Auto-PGD with CE
        loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks
        for APGD-CE include ART and Torchattacks. Notably, the Torchattacks
        implementation assumes an image (batches, channels, width, height).

        :return: APGD-CE adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (model, eps, eps_step, max_iter, nb_random_init, rho) = (
            self.atk_params["model"],
            self.linf,
            self.atk_params["alpha"],
            self.atk_params["epochs"],
            self.attacks["apgdce"].params["num_restarts"],
            self.attacks["apgdce"].traveler.optimizer.param_groups[0]["rho"],
        )
        art_adv = ta_adv = None
        if "art" in self.available:
            from art.attacks.evasion import AutoProjectedGradientDescent

            print("Producing APGD-CE adversarial examples with ART...")
            self.reset_seeds()
            art_adv = (
                torch.from_numpy(
                    AutoProjectedGradientDescent(
                        estimator=self.art_classifier,
                        norm="inf",
                        eps=eps,
                        eps_step=eps_step,
                        max_iter=max_iter,
                        targeted=False,
                        nb_random_init=nb_random_init,
                        batch_size=self.x.size(0),
                        loss_type="cross_entropy",
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        if (
            "torchattacks" in self.available
            and "shape" in self.atk_params["model"].params
        ):
            from torchattacks import APGD

            print("Producing APGD-CE adversarial examples with Torchattacks...")
            self.reset_seeds()
            ta_x = self.x.clone().unflatten(1, self.atk_params["model"].params["shape"])
            ta_adv = (
                APGD(
                    model=model,
                    norm="Linf",
                    eps=eps,
                    steps=max_iter,
                    n_restarts=nb_random_init,
                    seed=self.seed,
                    loss="ce",
                    eot_iter=1,
                    rho=rho,
                    verbose=self.verbose,
                )(inputs=ta_x, labels=self.y,).flatten(1),
                "Torchattacks",
            ).detach()
        self.reset_seeds()
        return self.attacks["apgdce"], tuple(
            fw for fw in (art_adv, ta_adv) if fw is not None
        )

    def apgddlr(self):
        """
        This method crafts adversarial examples with APGD-DLR (Auto-PGD with
        DLR loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported
        frameworks for APGD-DLR include ART and Torchattacks. Notably, DLR loss
        is undefined for these frameworks when there are only two classes.
        Moreover, the Torchattacks implementation assumes an image (batches,
        channels, width, height).

        :return: APGD-DLR adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (model, eps, eps_step, max_iter, nb_random_init, rho) = (
            self.atk_params["model"],
            self.linf,
            self.atk_params["alpha"],
            self.atk_params["epochs"],
            self.attacks["apgddlr"].params["num_restarts"],
            self.attacks["apgddlr"].traveler.optimizer.param_groups[0]["rho"],
        )
        art_adv = ta_adv = None
        if "art" in self.available and self.art_classifier.nb_classes > 2:
            from art.attacks.evasion import AutoProjectedGradientDescent

            print("Producing APGD-DLR adversarial examples with ART...")
            self.reset_seeds()
            art_adv = (
                torch.from_numpy(
                    AutoProjectedGradientDescent(
                        estimator=self.art_classifier,
                        norm="inf",
                        eps=eps,
                        eps_step=eps_step,
                        max_iter=max_iter,
                        targeted=False,
                        nb_random_init=nb_random_init,
                        batch_size=self.x.size(0),
                        loss_type="difference_logits_ratio",
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        if (
            "torchattacks" in self.available
            and self.art_classifier.nb_classes > 2
            and "shape" in self.atk_params["model"].params
        ):
            from torchattacks import APGD

            print("Producing APGD-DLR adversarial examples with Torchattacks...")
            self.reset_seeds()
            ta_x = self.x.clone().unflatten(1, self.atk_params["model"].params["shape"])
            ta_adv = (
                APGD(
                    model=model,
                    norm="Linf",
                    eps=eps,
                    steps=max_iter,
                    n_restarts=nb_random_init,
                    seed=self.seed,
                    loss="dlr",
                    eot_iter=1,
                    rho=rho,
                    verbose=self.verbose,
                )(inputs=ta_x, labels=self.y).flatten(1),
                "Torchattacks",
            ).detach()
        self.reset_seeds()
        return self.attacks["apgddlr"], tuple(
            fw for fw in (art_adv, ta_adv) if fw is not None
        )

    def bim(self):
        """
        This method crafts adversarial examples with BIM (Basic Iterative
        Method) (https://arxiv.org/pdf/1611.01236.pdf). The supported
        frameworks for BIM include AdverTorch, ART, CleverHans, Foolbox, and
        Torchattacks. Notably, CleverHans does not have an explicit
        implementation of BIM, so we call PGD, but with random initialization
        disabled.

        :return: BIM adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (model, eps, nb_iter, eps_iter) = (
            self.atk_params["model"],
            self.linf,
            self.atk_params["epochs"],
            self.atk_params["alpha"],
        )
        at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
        if "advertorch" in self.available:
            from advertorch.attacks import LinfBasicIterativeAttack

            print("Producing BIM adversarial examples with AdverTorch...")
            at_adv = (
                LinfBasicIterativeAttack(
                    predict=model,
                    loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                    eps=eps,
                    nb_iter=nb_iter,
                    eps_iter=eps_iter,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                    targeted=False,
                ).perturb(x=self.x.clone(), y=self.y.clone()),
                "AdverTorch",
            )
        if "art" in self.available:
            from art.attacks.evasion import BasicIterativeMethod

            print("Producing BIM adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    BasicIterativeMethod(
                        estimator=self.art_classifier,
                        eps=eps,
                        eps_step=eps_iter,
                        max_iter=nb_iter,
                        targeted=False,
                        batch_size=self.x.size(0),
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.projected_gradient_descent import (
                projected_gradient_descent as basic_iterative_method,
            )

            print("Producing BIM adversarial examples with CleverHans...")
            ch_adv = (
                basic_iterative_method(
                    model_fn=model,
                    x=self.x.clone(),
                    eps=eps,
                    eps_iter=eps_iter,
                    nb_iter=nb_iter,
                    norm=float("inf"),
                    clip_min=self.clip_min.max(),
                    clip_max=self.clip_max.min(),
                    y=self.y,
                    targeted=False,
                    rand_init=False,
                    rand_minmax=0,
                    sanity_checks=False,
                ).detach(),
                "CleverHans",
            )
        if "foolbox" in self.available:
            from foolbox.attacks import LinfBasicIterativeAttack

            print("Producing BIM adversarial examples with Foolbox...")
            _, fb_adv, _ = LinfBasicIterativeAttack(
                rel_stepsize=None,
                abs_stepsize=eps_iter,
                steps=nb_iter,
                random_start=False,
            )(
                self.fb_classifier,
                self.x.clone(),
                self.y.clone(),
                epsilons=eps,
            )
            fb_adv = (fb_adv, "Foolbox")
        if "torchattacks" in self.available:
            from torchattacks import BIM

            print("Producing BIM adversarial examples with Torchattacks...")
            ta_adv = (
                BIM(
                    model=model,
                    eps=eps,
                    alpha=eps_iter,
                    steps=nb_iter,
                )(inputs=self.x.clone(), labels=self.y),
                "Torchattacks",
            )
        return self.attacks["bim"], tuple(
            fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None
        )

    def cwl2(self):
        """
        This method crafts adversariale examples with CW-L2 (Carlini-Wagner
        with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported
        frameworks for CW-L2 include AdverTorch, ART, CleverHans, Foolbox, and
        Torchattacks. Notably, Torchattacks does not explicitly support binary
        searching on c (it expects searching manually).

        :return: CW-L2 adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (
            model,
            classes,
            confidence,
            learning_rate,
            binary_search_steps,
            max_iterations,
            initial_const,
        ) = (
            self.atk_params["model"],
            self.atk_params["model"].params["classes"],
            self.attacks["cwl2"].surface.loss.k,
            self.atk_params["alpha"],
            self.attacks["cwl2"].hparam_steps,
            self.attacks["cwl2"].epochs,
            self.attacks["cwl2"].surface.loss.c.item(),
        )
        at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
        if "advertorch" in self.available:
            from advertorch.attacks import CarliniWagnerL2Attack

            print("Producing CW-L2 adversarial examples with AdverTorch...")
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
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                ).perturb(x=self.x.clone(), y=self.y.clone()),
                "AdverTorch",
            )
        if "art" in self.available:
            from art.attacks.evasion import CarliniL2Method

            print("Producing CW-L2 adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    CarliniL2Method(
                        classifier=self.art_classifier,
                        confidence=confidence,
                        targeted=False,
                        learning_rate=learning_rate,
                        binary_search_steps=binary_search_steps,
                        max_iter=max_iterations,
                        initial_const=initial_const,
                        max_halving=binary_search_steps // 2,
                        max_doubling=binary_search_steps // 2,
                        batch_size=self.x.size(0),
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

            print("Producing CW-L2 adversarial examples with CleverHans...")
            ch_adv = (
                carlini_wagner_l2(
                    model_fn=model,
                    x=self.x.clone(),
                    n_classes=classes,
                    y=self.y,
                    lr=learning_rate,
                    confidence=confidence,
                    clip_min=self.clip_min.max(),
                    clip_max=self.clip_max.min(),
                    initial_const=initial_const,
                    binary_search_steps=binary_search_steps,
                    max_iterations=max_iterations,
                ).detach(),
                "CleverHans",
            )
        if "foolbox" in self.available:
            from foolbox.attacks import L2CarliniWagnerAttack

            print("Producing CW-L2 adversarial examples with Foolbox...")
            _, fb_adv, _ = L2CarliniWagnerAttack(
                binary_search_steps=binary_search_steps,
                steps=max_iterations,
                stepsize=learning_rate,
                confidence=confidence,
                initial_const=initial_const,
                abort_early=True,
            )(
                self.fb_classifier,
                self.x.clone(),
                self.y.clone(),
                epsilons=self.l2,
            )
            fb_adv = (fb_adv, "Foolbox")
        if "torchattacks" in self.available:
            from torchattacks import CW

            print("Producing CW-L2 adversarial examples with Torchattacks...")
            ta_adv = (
                CW(
                    model=model,
                    c=initial_const,
                    kappa=confidence,
                    steps=max_iterations,
                    lr=learning_rate,
                )(inputs=self.x.clone(), labels=self.y),
                "Torchattacks",
            )
        return self.attacks["cwl2"], tuple(
            fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None
        )

    def df(self):
        """
        This method crafts adversarial examples with DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
        include ART, Foolbox, and Torchattacks. Notably, the DF implementation
        in ART, Foolbox, and Torchattacks have an overshoot parameter which we
        set to aml DF's learning rate alpha minus one (not to be confused with
        the epsilon parameter used in aml, which governs the norm-ball size).

        :return: DeepFool adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        model, max_iter, epsilon, nb_grads, epsilons = (
            self.atk_params["model"],
            self.atk_params["epochs"],
            self.attacks["df"].params["α"] - 1,
            self.atk_params["model"].params["classes"],
            self.l2,
        )
        art_adv = fb_adv = ta_adv = None
        if "art" in self.available:
            from art.attacks.evasion import DeepFool

            print("Producing DF adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    DeepFool(
                        classifier=self.art_classifier,
                        max_iter=max_iter,
                        epsilon=epsilon,
                        nb_grads=nb_grads,
                        batch_size=self.x.size(0),
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        if "foolbox" in self.available:
            from foolbox.attacks import L2DeepFoolAttack

            print("Producing DF adversarial examples with Foolbox...")
            _, fb_adv, _ = L2DeepFoolAttack(
                steps=max_iter,
                candidates=nb_grads,
                overshoot=epsilon,
                loss="logits",
            )(
                self.fb_classifier,
                self.x.clone(),
                self.y.clone(),
                epsilons=epsilons,
            )
            fb_adv = (fb_adv, "Foolbox")
        if "torchattacks" in self.available:
            from torchattacks import DeepFool

            print("Producing DF adversarial examples with Torchattacks...")
            ta_adv = DeepFool(
                model=model,
                steps=max_iter,
                overshoot=epsilon,
            )(inputs=self.x.clone(), labels=self.y)

            # torchattack's implementation can return nans
            ta_adv = (torch.where(ta_adv.isnan(), self.x, ta_adv), "Torchattacks")
        return self.attacks["df"], tuple(
            fw for fw in (art_adv, fb_adv, ta_adv) if fw is not None
        )

    def fab(self):
        """
        This method crafts adversarial examples with FAB (Fast Adaptive
        Boundary) (https://arxiv.org/pdf/1907.02044.pdf). The supported
        frameworks for FAB include AdverTorch and Torchattacks.

        :return: FAB adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (model, n_restarts, n_iter, eps, alpha, eta, beta, n_classes) = (
            self.atk_params["model"],
            self.attacks["fab"].params["num_restarts"],
            self.atk_params["epochs"],
            self.l2,
            self.attacks["fab"].traveler.optimizer.param_groups[0]["alpha_max"],
            self.attacks["fab"].params["attack"].params["α"],
            self.attacks["fab"].traveler.optimizer.param_groups[0]["beta"],
            self.atk_params["model"].params["classes"],
        )
        at_adv = ta_adv = None
        if "advertorch" in self.available:
            from advertorch.attacks import L2FABAttack

            print("Producing FAB adversarial examples with AdverTorch...")
            self.reset_seeds()
            at_adv = (
                L2FABAttack(
                    predict=model,
                    n_restarts=n_restarts,
                    n_iter=n_iter,
                    eps=eps,
                    alpha_max=alpha,
                    eta=eta,
                    beta=beta,
                    verbose=self.verbose,
                ).perturb(x=self.x.clone(), y=self.y.clone()),
                "AdverTorch",
            )
        if "torchattacks" in self.available:
            from torchattacks import FAB

            print("Producing FAB adversarial examples with Torchattacks...")
            self.reset_seeds()
            ta_adv = (
                FAB(
                    model=model,
                    norm="L2",
                    eps=eps,
                    steps=self.atk_params["epochs"],
                    n_restarts=n_restarts,
                    alpha_max=alpha,
                    eta=eta,
                    beta=beta,
                    verbose=self.verbose,
                    seed=self.seed,
                    multi_targeted=False,
                    n_classes=n_classes,
                )(inputs=self.x.clone(), labels=self.y),
                "Torchattacks",
            )
        self.reset_seeds()
        return self.attacks["fab"], tuple(
            fw for fw in (at_adv, ta_adv) if fw is not None
        )

    def jsma(self):
        """
        This method crafts adversarial examples with JSMA (Jacobian Saliency
        Map Approach) (https://arxiv.org/pdf/1511.07528.pdf). The supported
        frameworks for the JSMA include AdverTorch and ART. Notably, the JSMA
        implementation in AdverTorch and ART both assume the l0-norm is passed
        in as a percentage (which is why we pass in linf) and we set theta to
        be 1 since features can only be perturbed once. Moreover, the
        AdverTorch implementation does not suport an untargetted scheme, so we
        supply random targets.

        :return: JSMA adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (model, num_classes, gamma, theta) = (
            self.atk_params["model"],
            self.atk_params["model"].params["classes"],
            self.linf,
            1,
        )
        at_adv = art_adv = None
        if "advertorch" in self.available:
            from advertorch.attacks import JacobianSaliencyMapAttack

            print("Producing JSMA adversarial examples with AdverTorch...")
            at_adv = (
                JacobianSaliencyMapAttack(
                    predict=model,
                    num_classes=num_classes,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                    gamma=gamma,
                    theta=theta,
                ).perturb(
                    x=self.x.clone(), y=torch.randint(num_classes, self.y.size())
                ),
                "AdverTorch",
            )
        if "art" in self.available:
            from art.attacks.evasion import SaliencyMapMethod

            print("Producing JSMA adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    SaliencyMapMethod(
                        classifier=self.art_classifier,
                        theta=theta,
                        gamma=gamma,
                        batch_size=self.x.size(0),
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["jsma"], tuple(
            fw for fw in (at_adv, art_adv) if fw is not None
        )

    def pgd(self):
        """
        This method crafts adversarial examples with PGD (Projected Gradient
        Descent)) (https://arxiv.org/pdf/1706.06083.pdf). The supported
        frameworks for PGD include AdverTorch, ART, CleverHans, and
        Torchattacks.

        :return: PGD adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        (model, eps, nb_iter, eps_iter) = (
            self.atk_params["model"],
            self.linf,
            self.atk_params["epochs"],
            self.atk_params["alpha"],
        )
        at_adv = art_adv = ch_adv = fb_adv = ta_adv = None
        if "advertorch" in self.available:
            from advertorch.attacks import LinfPGDAttack

            print("Producing PGD adversarial examples with AdverTorch...")
            self.reset_seeds()
            at_adv = (
                LinfPGDAttack(
                    predict=model,
                    loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                    eps=eps,
                    nb_iter=nb_iter,
                    eps_iter=eps_iter,
                    rand_init=True,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                    targeted=False,
                ).perturb(x=self.x.clone(), y=self.y.clone()),
                "AdverTorch",
            )
        if "art" in self.available:
            from art.attacks.evasion import ProjectedGradientDescent

            print("Producing PGD adversarial examples with ART...")
            self.reset_seeds()
            art_adv = (
                torch.from_numpy(
                    ProjectedGradientDescent(
                        estimator=self.art_classifier,
                        norm="inf",
                        eps=eps,
                        eps_step=eps_iter,
                        decay=None,
                        max_iter=nb_iter,
                        targeted=False,
                        num_random_init=1,
                        batch_size=self.x.size(0),
                        random_eps=True,
                        summary_writer=False,
                        verbose=self.verbose,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.projected_gradient_descent import (
                projected_gradient_descent,
            )

            print("Producing PGD adversarial examples with CleverHans...")
            self.reset_seeds()
            ch_adv = (
                projected_gradient_descent(
                    model_fn=model,
                    x=self.x.clone(),
                    eps=eps,
                    eps_iter=eps_iter,
                    nb_iter=nb_iter,
                    norm=float("inf"),
                    clip_min=self.clip_min.max(),
                    clip_max=self.clip_max.min(),
                    y=self.y,
                    targeted=False,
                    rand_init=True,
                    rand_minmax=eps,
                    sanity_checks=False,
                ).detach(),
                "CleverHans",
            )
        if "foolbox" in self.available:
            from foolbox.attacks import LinfProjectedGradientDescentAttack

            print("Producing PGD adversarial examples with Foolbox...")
            self.reset_seeds()
            _, fb_adv, _ = LinfProjectedGradientDescentAttack(
                rel_stepsize=None,
                abs_stepsize=eps_iter,
                steps=nb_iter,
                random_start=True,
            )(
                self.fb_classifier,
                self.x.clone(),
                self.y.clone(),
                epsilons=eps,
            )
            fb_adv = (fb_adv, "Foolbox")
        if "torchattacks" in self.available:
            from torchattacks import PGD

            print("Producing PGD adversarial examples with Torchattacks...")
            self.reset_seeds()
            ta_adv = (
                PGD(
                    model=model,
                    eps=eps,
                    alpha=eps_iter,
                    steps=nb_iter,
                    random_start=True,
                )(inputs=self.x.clone(), labels=self.y),
                "Torchattacks",
            )
        self.reset_seeds()
        return self.attacks["pgd"], tuple(
            fw for fw in (at_adv, art_adv, ch_adv, fb_adv, ta_adv) if fw is not None
        )


class FunctionalTests(BaseTest):
    """
    The following class implements functional tests. Functional correctness
    tests involve crafting adversarial examples and verifying that model
    accuracy can be dropped to <1% for some "small" lp-budget. This is
    typically defined as ~15% consumption of budget measured by the target
    lp-norm (e.g., for a space with 100 features, this is budget is defined as
    15 in l0, 10 in l2, and 0.15 in l∞).

    The following attacks are supported:
        APGD-CE (Auto-PGD with CE loss) (https://arxiv.org/pdf/2003.01690.pdf)
        APGD-DLR (Auto-PGD with DLR loss) (https://arxiv.org/pdf/2003.01690.pdf)
        BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
        CW-L2 (Carlini-Wagner with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf)
        DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
        FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
        JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)
        PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)

    :func:`functional_test`: performs a functional test
    :func:`setUpClass`: sets functional test parameters
    :func:`test_apgdce`: functional test for APGD-CE
    :func:`test_apgddlr`: functional test for APGD-DLR
    :func:`test_bim`: functional test for BIM
    :func:`test_cwl2`: functional test for CW-L2
    :func:`test_df`: functional test for DF
    :func:`test_fab`: functional test for FAB
    :func:`test_jsma`: functional test for JSMA
    :func:`test_pgd`: functional test for PGD
    """

    @classmethod
    def setUpClass(cls, min_acc=0.01):
        """
        This method initializes the functional testing framework by setting
        parameters unique to the tests within this test case. At this time,
        this just sets the minimum accuracy needed for an attack to be
        considered "functionally" correct (notably, this should be tuned
        appropriately with the number of epochs and norm-ball size defined in
        the setUpModule function).

        :param min_acc: the minimum accuracy necessary for a "successful" attack
        :type min_acc: float
        :return: None
        :rtype: NoneType
        """
        super().setUpClass()
        cls.min_acc = min_acc
        return None

    def functional_test(self, attack):
        """
        This method performs a functional test for a given attack.

        :param adversary: adversary to test
        :type adversary: aml Adversary or Attack object
        :return: None
        :rtype: NoneType
        """
        with self.subTest(Attack=attack.name):
            p = attack.craft(self.x, self.y)
            norms = (p.norm(d, 1).mean().item() for d in (0, 2, torch.inf))
            norm_results = ", ".join(
                f"l{p}: {n:.3}/{b} ({n/b:.2%})"
                for n, b, p in zip(norms, (self.l0, self.l2, self.linf), (0, 2, "∞"))
            )
            adv_acc = self.model.accuracy(self.x + p, self.y).item()
            print(f"{attack.name} complete! Model Acc: {adv_acc:.2%},", norm_results)
            self.assertLess(adv_acc, self.min_acc)
        return None

    def test_apgdce(self):
        """
        This method performs a functional test for APGD-CE (Auto-PGD with CE
        loss) (https://arxiv.org/pdf/2003.01690.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["apgdce"])

    def test_apgddlr(self):
        """
        This method performs a functional test for APGD-DLR (Auto-PGD with DLR
        loss) (https://arxiv.org/pdf/2003.01690.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["apgddlr"])

    def test_bim(self):
        """
        This method performs a functional test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["bim"])

    def test_cwl2(self):
        """
        This method performs a functional test for CW-L2 (Carlini-Wagner with
        l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["cwl2"])

    def test_df(self):
        """
        This method performs a functional test for DF (DeepFool)
        (https://arxiv.org/pdf/1511.04599.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["df"])

    def test_fab(self):
        """
        This method performs a functional test for FAB (Fast Adaptive Boundary)
        (https://arxiv.org/pdf/1907.02044.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["fab"])

    def test_jsma(self):
        """
        This method performs a functional test for JSMA (Jacobian Saliency Map
        Approach) (https://arxiv.org/pdf/1511.07528.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["jsma"])

    def test_pgd(self):
        """
        This method performs a functional test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["pgd"])


class IdentityTests(BaseTest):
    """
    The following class implements identity tests. Identity correctness tests
    assert that the feature values of aml adversarial examples themselves must
    be at least 99% similar to adversarial examples of other frameworks.
    Feature values are considered similar if their difference satifies:

                        |other - aml| ≤ atol + rtol * |aml|

    where other are the perturbations of other frameworks, aml is the
    perturbations of the aml framework, atol and rtol are absolute and relative
    tolerance, respectively. The values atol and rtol take depends on the
    datatype of the underlying tensors, as described in
    https://pytorch.org/docs/stable/testing.html. Notably, aml implementations
    deviate from other libraries in many ways, and thus a only a limited set of
    attacks are expected to pass.

    The following frameworks are supported:
        AdverTorch (https://github.com/BorealisAI/advertorch)
        ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
        CleverHans (https://github.com/cleverhans-lab/cleverhans)
        Foolbox (https://github.com/bethgelab/foolbox)
        Torchattacks (https://github.com/Harry24k/adversarial-attacks-pytorch)

    The following attacks are supported:
        BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
        DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
        PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)

    :func:`identity_test`: performs an identity test
    :func:`setUpClass`: sets identity test parameters
    :func:`test_apgdce`: identity test for APGD-CE
    :func:`test_apgddlr`: identity test for APGD-DLR
    :func:`test_bim`: identity test for BIM
    :func:`test_cwl2`: identity test for CW-L2
    :func:`test_df`: identity test for DF
    :func:`test_fab`: identity test for FAB
    :func:`test_jsma`: identity test for JSMA
    :func:`test_pgd`: identity test for PGD
    """

    @classmethod
    def setUpClass(cls, max_distance=0.001):
        """
        This method initializes the identity testing case.

        :param max_distance: the maximum allowable distance between features
        :type max_distance: float
        :return: None
        :rtype: NoneType
        """
        super().setUpClass()
        cls.max_distance = max_distance
        return None

    def identity_test(self, attack, fws):
        """
        This method performs an identity test for a given attack.

        :param attack: attack to test
        :type attack: aml Attack object
        :param fws: adversarial examples produced by other frameworks
        :type fws: tuple of tuples of torch Tensor object (n, m) and str
        :return: None
        :rtype: NoneType
        """

        # craft adversarial examples
        fws_adv, fws = zip(*fws) if fws else ([], "No Adversarial Examples")
        aml_p = attack.craft(self.x, self.y)
        fws_p = (fw_adv.sub(self.x) for fw_adv in fws_adv)

        # compute perturbation differences
        diffs = tuple(aml_p.sub(fw_p).abs() for fw_p in fws_p)

        # compute interesting statistics
        diff_stats = (
            (diff.min(), diff.max(), diff.mean(), diff.median(), diff.std())
            for diff in diffs
        )

        # print results and assert adversarial example similarity
        for (mn, mx, me, md, st), f in zip(diff_stats, fws):
            print(
                f"{attack.name} aml v. {f}: Min: {mn:.3}, Max {mx:.3},",
                f"Mean: {me:.3}, Median: {md:.3} SD: {st:.3}",
            )
        rtol, atol = torch.testing._comparison.default_tolerances(*diffs)
        for fw, diff in zip(fws, diffs):
            with self.subTest(Attack=f"{attack.name} v. {fw}"):
                close = diff.isclose(torch.zeros_like(diff), atol, rtol)
                print(
                    f"{close.all(1).sum().item()}/{close.size(0)}",
                    f"({close.all(1).sum().div(close.size(0)).item():.2%})",
                    f"of {attack.name} aml adversarial examples are close to {fw}.",
                )

                # if the identity test fails, aim to debug why
                org_msg = torch.testing._comparison.make_tensor_mismatch_msg(
                    diff, torch.zeros_like(diff), ~close, rtol=rtol, atol=atol
                )
                targets = close.all(1) == self.model.correct
                targets_msg = (
                    "Percentage of failed inputs that were initially misclassified: "
                    f"{targets.sum().div(targets.numel()):.2%}"
                )
                self.assertIsNone(
                    torch.testing.assert_close(
                        diff,
                        torch.zeros_like(diff),
                        msg="\n".join((org_msg, targets_msg)),
                    )
                )
        return None

    def test_bim(self):
        """
        This method performs an identity test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf). Notably, the implementation of
        BIM in ART ostensibly always fails, given that ART always uses the
        current model logits in untargetted attacks and thus, *corrects*
        misclassified inputs such that they are classified correctly. In
        effect, the difference between ART and aml adversarial examples from
        BIM is commonly the linf threat model, epsilon. However, when
        minimizing model accuracy (instead of maximizing model loss), the
        identity check should succeed, given that initially misclassified
        inputs are skipped.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.bim())

    def test_df(self):
        """
        This method performs an identity test for DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf).
        :return: None
        :rtype: NoneType
        """
        attack, fws = self.df()
        return self.identity_test(*self.df())

    def test_pgd(self):
        """
        This method performs an identity test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf). Notably, the
        implementation of PGD in ART will always fail, given that the current
        implementation initializes the attaack by drawing from a truncated
        normal distribution, instead of from a uniform distribution. The
        motivation for this difference in initialization is sourced by
        https://arxiv.org/pdf/1611.01236.pdf to assist FGSM-based adversarial
        training in generalizing across different epsilons. ART claims that the
        effectiveness of this method is untested with PGD.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.pgd())


class SemanticTests(BaseTest):
    """
    The following class implements semantic tests. Semantic correctness tests
    are more sophisticated than functional tests, in that they compare the the
    performance of adversarial examples produced by known attacks within aml to
    those implemented in other adversarial machine learning frameworks. Attacks
    in aml are determined to be semantically correct if the produced
    adversarial examples are within no worse than 1% of the performance of
    adversarial examples produced by other frameworks. Performance is defined
    as one minus the l2-norm of normalized decrease in model accuracy and
    normalized lp-norm.

    The following frameworks are supported:
        AdverTorch (https://github.com/BorealisAI/advertorch)
        ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
        CleverHans (https://github.com/cleverhans-lab/cleverhans)
        Foolbox (https://github.com/bethgelab/foolbox)
        Torchattacks (https://github.com/Harry24k/adversarial-attacks-pytorch)

    The following attacks are supported:
        APGD-CE (Auto-PGD with CE loss) (https://arxiv.org/pdf/2003.01690.pdf)
        APGD-DLR (Auto-PGD with DLR loss) (https://arxiv.org/pdf/2003.01690.pdf)
        BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
        CW-L2 (Carlini-Wagner with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf)
        DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
        FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
        JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)
        PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)

    :func:`semantic_test`: performs a semantic test
    :func:`setUpClass`: sets semantic test parameters
    :func:`test_apgdce`: semantic test for APGD-CE
    :func:`test_apgddlr`: semantic test for APGD-DLR
    :func:`test_bim`: semantic test for BIM
    :func:`test_cwl2`: semantic test for CW-L2
    :func:`test_df`: semantic test for DF
    :func:`test_fab`: semantic test for FAB
    :func:`test_jsma`: semantic test for JSMA
    :func:`test_pgd`: semantic test for PGD
    """

    @classmethod
    def setUpClass(cls, max_perf_degrad=0.01):
        """
        This method initializes the semantic testing case by setting parameters
        unique to the tests within this test case. At this time, this sets the
        maximum allowable performance degration between adversarial examples
        produced by aml and other frameworks such that the aml attacks are
        considered to be "semantically" correct. Performance (denoted as 𝒫) is
        measured as the one minus the l2-norm of normalized decrease in model
        accuracy and normalized lp-norm, where l0 is normalized by the total
        number of features, l2 by the square root of the total number of
        features, and l∞ need not be normalized since, during crafting,
        feautres are always between 0 and 1. Notably, if aml attacks are
        *better* than other frameworks by greater than the maximum allowable
        performance degradation, the tests still pass (failure can only occurs
        if aml attacks are worse). Finally, this method also instantiates an
        ART PyTorch classifier to be used by all ART-based attacks.

        :param max_perf_degrad: the maximum allowable difference in performance
        :type max_perf_degrad: float
        :return: None
        :rtype: NoneType
        """
        super().setUpClass()
        cls.max_perf_degrad = max_perf_degrad
        cls.state = 0
        return None

    def semantic_test(self, attack, fws):
        """
        This method performs a semantic test for a given attack.

        :param adversary: attack to test
        :type adversary: aml Adversary object
        :param fws: adversarial examples produced by other frameworks
        :type fws: tuple of tuples of torch Tensor object (n, m) and str
        :return: None
        :rtype: NoneType
        """

        # craft adversarial examples
        fws_adv, fws = zip(*fws) if fws else ([], "No Adversarial Examples")
        aml_p = attack.craft(self.x, self.y)
        fws_p = (fw_adv.sub(self.x) for fw_adv in fws_adv)

        # project to threat model, clip to domain, and compute norms
        lp = (0, 2, torch.inf)
        proj = {0: self.l0_proj, 2: self.l2_proj, torch.inf: self.linf_proj}[attack.lp]
        ps = tuple(proj(p).clamp(self.p_min, self.p_max) for p in (aml_p, *fws_p))
        norms = tuple([p.norm(d, 1).mean().item() for d in lp] for p in ps)

        # compute model accuracy decrease
        advs = (self.x + p for p in ps)
        acc_abs = tuple(self.model.accuracy(adv, self.y).item() for adv in advs)
        acc_dec = tuple(a / self.clean_acc for a in acc_abs)

        # compute perturbation budget consumption
        max_p = (self.l0_max, self.l2_max, self.linf_max)
        atk_b = ((n / b for n, b in zip(an, max_p)) for an in norms)

        # compute performance and organize printing results
        max_b = (self.l0, self.l2, self.linf)
        perfs = tuple(
            [max(1 - (aa**2 + b**2) ** (1 / 2), 0) for b in ab]
            for ab, aa in zip(atk_b, acc_dec)
        )
        norms_perfs = (
            ", ".join(
                f"l{p}: {n:.3}/{float(b):.3} ({n/b:.2%}) 𝒫: {f:.1%}"
                for p, n, b, f in zip((0, 2, "∞"), an, max_b, perf)
            )
            for an, perf in zip(norms, perfs)
        )
        acc_diff = (a - self.clean_acc for a in acc_abs)
        lf = max((len(f) for f in ("aml", *fws)))
        results = zip(("aml", *fws), acc_abs, acc_diff, norms_perfs)
        for f, a, d, n in results:
            print(f"{f:>{lf}} {attack.name} Model Acc: {a:.2%} ({d:.2%}), Results: {n}")

        # compute target norm and assert marginal performance difference
        norm_map = (aml.surface.l0, aml.surface.l2, aml.surface.linf)
        atk_norm = norm_map.index(attack.surface.norm)
        norm_perfs = (p[atk_norm] for p in perfs)
        aml_perf, fws_perf = next(norm_perfs), tuple(norm_perfs)
        for fw, fw_perf in zip(fws, fws_perf):
            with self.subTest(Attack=f"{attack.name} v. {fw}"):
                self.assertGreaterEqual(aml_perf + self.max_perf_degrad, fw_perf)
        return None

    def test_apgdce(self):
        """
        This method performs a semantic test for APGD-CE (Auto-PGD with CE
        loss) (https://arxiv.org/pdf/2003.01690.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.apgdce())

    def test_apgddlr(self):
        """
        This method performs a semantic test for APGD-DLR (Auto-PGD with DLR
        loss) (https://arxiv.org/pdf/2003.01690.pdf). Notably, DLR loss is
        undefined for other frameworks when there are only two classes.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.apgddlr())

    def test_bim(self):
        """
        This method performs a semantic test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.bim())

    def test_cwl2(self):
        """
        This method performs a semantic test for CW-L2 (Carlini-Wagner with l₂
        norm) (https://arxiv.org/pdf/1608.04644.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.cwl2())

    def test_df(self):
        """
        This method performs a semantic test for DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.df())

    def test_fab(self):
        """
        This method performs a semantic test for FAB (Fast Adaptive Boundary)
        (https://arxiv.org/pdf/1907.02044.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.fab())

    def test_jsma(self):
        """
        This method performs a semantic test for JSMA (Jacobian Saliency Map
        Approach) (https://arxiv.org/pdf/1511.07528.pdf). Notably, the JSMA
        traditionally can only perturb a feature once, we override the value of
        alpha to be 1 so that tests are passed.

        :return: None
        :rtype: NoneType
        """
        attack, fws = self.jsma()
        return self.semantic_test(
            aml.attacks.jsma(**self.atk_params | {"alpha": 1, "epsilon": self.l0}),
            fws,
        )

    def test_pgd(self):
        """
        This method performs a semantic test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.pgd())


class SpecialTests(BaseTest):
    """
    The following class implements the special test case. Special tests are
    designed to exercise components of the Space of Adversarial Strategies
    framework (https://arxiv.org/pdf/2209.04521.pdf) in specific ways. The
    purpose and detailed description of each test can be found in their
    respecitve methods.

    :func:`test_all_components`: severe component coverage test
    :func:`test_known_attacks`: moderate component coverage test
    """

    @classmethod
    def setUpClass(cls):
        """
        This method initializes the special testing framework. Given the naturally
        unique nature of special tests, it accepts no arguments.

        :return: None
        :rtype: NoneType
        """
        super().setUpClass()
        return None

    def test_all_components(self, samples=10, epochs=3):
        """
        This method is a heavyweight coverage test. It calls the attack builder
        function in the attacks module to instantiate 2x every possible
        combination of attacks (one set corresponding to max-loss adversaries
        and the other corresponding to min-accuracy adversaries). Naturally, to
        make this a reasonable test to run regularly, we attack a handful
        samples for a couple iterations, per attack. Notably, the Adversary
        layer is not tested (that is, no restarts and no hyperparameter
        optimization). This test is considered successful if no exceptions are
        raised.

        :param samples: number of samples to attack
        :type samples: int
        :param epochs: number of attack iterations
        :type epochs: int
        :return: None
        :rtype: NoneType
        """
        alpha, epochs, model = (
            self.atk_params["alpha"],
            min(epochs, self.atk_params["epochs"]),
            self.atk_params["model"],
        )
        attacks = tuple(
            a
            for early_termination in (True, False)
            for epsilon, norm in zip(
                (self.l0, self.l2, self.linf),
                (aml.surface.l0, aml.surface.l2, aml.surface.linf),
            )
            for a in aml.attacks.attack_builder(
                alpha=alpha,
                epochs=epochs,
                early_termination=early_termination,
                epsilon=epsilon,
                model=model,
                norms=(norm,),
                verbosity=0,
            )
        )
        for i, attack in enumerate(attacks, start=1):
            with self.subTest(Attack=f"{i}: {attack.name}"):
                print(
                    f"Testing {attack.name:<10}..."
                    f"{i}/{len(attacks)} ({i / len(attacks):.1%})",
                    end="\r",
                )
                attack.craft(self.x[:samples], self.y[:samples])
        return None

    def test_known_components(self, samples=10, epochs=3):
        """
        This method is a moderat coverage test. It calls all of the known
        attack helper functions in the aml attacks, i.e., APGD-CE, APGD-DLR,
        BIM, CW-L2, DF, FAB, JSMA, and PGD. With these attacks, all components
        are individually tests (but complex interactions may be missed).
        Naturally, to make this a reasonable test to run regularly, we attack a
        handful of samples for a couple iterations, per attack. This test is
        considered successful if no exceptions are raised.

        :param samples: number of samples to attack
        :type samples: int
        :param epochs: number of attack iterations
        :type epochs: int
        :return: None
        :rtype: NoneType
        """
        alpha, epochs, model = (
            self.atk_params["alpha"],
            min(epochs, self.atk_params["epochs"]),
            self.atk_params["model"],
        )
        attacks = (
            aml.attacks.apgdce(alpha, epochs, self.linf, model, verbosity=0),
            aml.attacks.apgddlr(alpha, epochs, self.linf, model, verbosity=0),
            aml.attacks.bim(alpha, epochs, self.linf, model, verbosity=0),
            aml.attacks.cwl2(alpha, epochs, self.l2, model, verbosity=0),
            aml.attacks.df(alpha, epochs, self.l2, model, verbosity=0),
            aml.attacks.fab(alpha, epochs, self.linf, model, verbosity=0),
            aml.attacks.jsma(alpha, epochs, self.l0, model, verbosity=0),
            aml.attacks.pgd(alpha, epochs, self.linf, model, verbosity=0),
        )
        for i, attack in enumerate(attacks, start=1):
            with self.subTest(Attack=f"{i}: {attack.name}"):
                print(
                    f"Testing {attack.name:<10}..."
                    f"{i}/{len(attacks)} ({i / len(attacks):.1%})",
                    end="\r",
                )
                attack.craft(self.x[:samples], self.y[:samples])
        return None
