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
import unittest  # Unit testing framework
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# get all functional tests passing
# implement identity tests
# implement semantic tests
# implement special tests
# integrate binary search steps for aml cwl2 tests
# need a better measure of performance for semantic tests (linf always fails bc of budget)
# move identity test failure debug prints into assert msg (need to wrap torch assert into unittest assert)
# figure out how to get random seeds to be static across all other frameworks


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
        cls, alpha=0.01, dataset="phishing", debug=False, epochs=30, norm=0.15
    ):
        """
        This function initializes the setup necessary for all test cases within
        this module. Specifically, this method retrieves data, processes it,
        trains a model, instantiates attacks, imports availabe external
        libraries (used for semantic and identity tests), and loads PyTorch in
        debug mode if desired (to assist debugging autograd).

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
        :return: None
        :rtype: NoneType
        """

        # set debug mode, load data (extract training and test sets, if they exist)
        print("Initializing module for all test cases...")
        torch.autograd.set_detect_anomaly(debug)
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

        # determine clipping range (datasets need not be between 0 and 1)
        mins, idx = cls.x.min(0)
        maxs, idx = cls.x.max(0)
        clip = (mins, maxs)
        clip_info = (
            (mins[0].item(), maxs[0].item())
            if (mins[0].eq(mins).all() and maxs[0].eq(maxs).all())
            else f"(({mins.min().item()}, ..., {mins.max()}.item()),"
            "({maxs.min().item()}, ..., {maxs.max().item()}))"
        )

        # train model and save craftset clean accuracy (used for semantic tests)
        template = getattr(dlm.architectures, dataset)
        cls.model = (
            dlm.CNNClassifier(template=template)
            if template.CNNClassifier is not None
            else dlm.MLPClassifier(template=template)
        )
        cls.model.fit(*(x, y) if has_test else (cls.x, cls.y))
        cls.clean_acc = cls.model.accuracy(cls.x, cls.y).item()

        # instantiate attacks and save attack parameters
        cls.l0 = int(cls.x.size(1) * norm)
        cls.l2 = maxs.sub(mins).norm(2).item()
        cls.linf = norm
        cls.attack_params = {
            "alpha": alpha,
            "clip": clip,
            "epochs": epochs,
            "model": cls.model,
        }
        cls.attacks = {
            "apgdce": aml.attacks.apgdce(**cls.attack_params | {"epsilon": cls.linf}),
            "apgddlr": aml.attacks.apgddlr(**cls.attack_params | {"epsilon": cls.linf}),
            "bim": aml.attacks.bim(**cls.attack_params | {"epsilon": cls.linf}),
            "cwl2": aml.attacks.cwl2(**cls.attack_params | {"epsilon": cls.l2}),
            "df": aml.attacks.df(**cls.attack_params | {"epsilon": cls.l2}),
            "fab": aml.attacks.fab(**cls.attack_params | {"epsilon": cls.l2}),
            "jsma": aml.attacks.jsma(**cls.attack_params | {"epsilon": cls.l0}),
            "pgd": aml.attacks.pgd(**cls.attack_params | {"epsilon": cls.linf}),
        }

        # determine available frameworks and set art classifier if needed
        supported = ("cleverhans", "art")
        cls.available = [f for f in supported if importlib.util.find_spec(f)]
        frameworks = ", ".join(cls.available)
        cls.art_classifier = (
            cls.build_art_classifier()
            if cls in (IdentityTests, SemanticTests)
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
            f"Attack Strength α: {cls.attack_params['alpha']}",
            f"Attack Epochs: {cls.attack_params['epochs']}",
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
            model=cls.attack_params["model"].model,
            clip_values=(
                cls.attack_params["clip"][0].max().item(),
                cls.attack_params["clip"][1].min().item(),
            ),
            loss=cls.attack_params["model"].loss,
            optimizer=cls.attack_params["model"].optimizer,
            input_shape=cls.x.shape,
            nb_classes=cls.attack_params["model"].params["classes"],
        )

    def apgdce(self):
        """
        This method crafts adversarial examples with APGD-CE (Auto-PGD with CE
        loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks
        for APGD-CE include ART.

        :return: APGD-CE adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        art_adv = None
        if "art" in self.available:
            from art.attacks.evasion import AutoProjectedGradientDescent

            print("Producing APGD-CE adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    AutoProjectedGradientDescent(
                        estimator=self.art_classifier,
                        norm="inf",
                        eps=self.linf,
                        eps_step=self.attack_params["alpha"],
                        max_iter=self.attack_params["epochs"],
                        targeted=False,
                        nb_random_init=1,
                        batch_size=self.x.size(0),
                        loss_type="cross_entropy",
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["apgdce"], tuple(fw for fw in (art_adv,) if fw is not None)

    def apgddlr(self):
        """
        This method crafts adversarial examples with APGD-DLR (Auto-PGD with
        DLR loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported
        frameworks for APGD-DLR include ART. Notably, DLR loss is undefined
        when there are only two classes.

        :return: APGD-DLR adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        art_adv = None
        if "art" in self.available and self.art_classifier.nb_classes > 2:
            from art.attacks.evasion import AutoProjectedGradientDescent

            print("Producing APGD-CE adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    AutoProjectedGradientDescent(
                        estimator=self.art_classifier,
                        norm="inf",
                        eps=self.linf,
                        eps_step=self.attack_params["alpha"],
                        max_iter=self.attack_params["epochs"],
                        targeted=False,
                        nb_random_init=1,
                        batch_size=self.x.size(0),
                        loss_type="difference_logits_ratio",
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["apgddlr"], tuple(fw for fw in (art_adv,) if fw is not None)

    def bim(self):
        """
        This method crafts adversarial examples with BIM (Basic Iterative
        Method) (https://arxiv.org/pdf/1611.01236.pdf). The supported
        frameworks for BIM include CleverHans and ART. Notably, CleverHans does
        not have an explicit implementation of BIM, so we call PGD, but with
        random initialization disabled.

        :return: BIM adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        ch_adv = art_adv = None
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.projected_gradient_descent import (
                projected_gradient_descent as basic_iterative_method,
            )

            print("Producing BIM adversarial examples with CleverHans...")
            ch_adv = (
                basic_iterative_method(
                    model_fn=self.attack_params["model"],
                    x=self.x.clone(),
                    eps=self.linf,
                    eps_iter=self.attack_params["alpha"],
                    nb_iter=self.attack_params["epochs"],
                    norm=float("inf"),
                    clip_min=self.attack_params["clip"][0].max().item(),
                    clip_max=self.attack_params["clip"][1].min().item(),
                    y=self.y,
                    targeted=False,
                    rand_init=False,
                    rand_minmax=0,
                    sanity_checks=True,
                ).detach(),
                "CleverHans",
            )
        if "art" in self.available:
            from art.attacks.evasion import BasicIterativeMethod

            print("Producing BIM adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    BasicIterativeMethod(
                        estimator=self.art_classifier,
                        eps=self.linf,
                        eps_step=self.attack_params["alpha"],
                        max_iter=self.attack_params["epochs"],
                        targeted=False,
                        batch_size=self.x.size(0),
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["bim"], tuple(
            fw for fw in (ch_adv, art_adv) if fw is not None
        )

    def cwl2(self):
        """
        This method crafts adversariale examples with CW-L2 (Carlini-Wagner
        with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported
        frameworks for CW-L2 include CleverHans and ART.

        :return: CW-L2 adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        ch_adv = art_adv = None
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

            print("Producing CW-L2 adversarial examples with CleverHans...")
            ch_adv = (
                carlini_wagner_l2(
                    model_fn=self.attack_params["model"],
                    x=self.x.clone(),
                    n_classes=self.attack_params["model"].params["classes"],
                    y=self.y,
                    lr=self.attack_params["alpha"],
                    confidence=self.attacks["cwl2"].surface.loss.k,
                    clip_min=self.attack_params["clip"][0].max().item(),
                    clip_max=self.attack_params["clip"][1].min().item(),
                    initial_const=self.attacks["cwl2"].surface.loss.c.item(),
                    binary_search_steps=1,
                    max_iterations=self.attack_params["epochs"],
                ).detach(),
                "CleverHans",
            )
        if "art" in self.available:
            from art.attacks.evasion import CarliniL2Method

            print("Producing CW-L2 adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    CarliniL2Method(
                        classifier=self.art_classifier,
                        confidence=self.attacks["cwl2"].surface.loss.k,
                        targeted=False,
                        learning_rate=self.attack_params["alpha"],
                        binary_search_steps=1,
                        max_iter=self.attack_params["epochs"],
                        initial_const=self.attacks["cwl2"].surface.loss.c.item(),
                        max_halving=5,
                        max_doubling=5,
                        batch_size=self.x.size(0),
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["cwl2"], tuple(
            fw for fw in (ch_adv, art_adv) if fw is not None
        )

    def df(self):
        """
        This method crafts adversarial examples with DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
        include ART. Notably, the DF implementation in ART has an overshoot
        parameter which we set to 0 (not to be confused with the epsilon
        parameter used in aml, which governs the norm-ball size).

        :return: DeepFool adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        art_adv = None
        if "art" in self.available:
            from art.attacks.evasion import DeepFool

            print("Producing DF adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    DeepFool(
                        classifier=self.art_classifier,
                        max_iter=self.attack_params["epochs"],
                        epsilon=0,
                        nb_grads=self.attack_params["model"].params["classes"],
                        batch_size=self.x.size(0),
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["df"], tuple(fw for fw in (art_adv,) if fw is not None)

    def fab(self):
        """
        This method crafts adversarial examples with FAB (Fast Adaptive
        Boundary) (https://arxiv.org/pdf/1907.02044.pdf). There are no
        currently supported frameworks.

        :return: FAB adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        return self.attacks["fab"], tuple(fw for fw in (None,) if fw is not None)

    def jsma(self):
        """
        This method crafts adversarial examples with JSMA (Jacobian Saliency
        Map Approach) (https://arxiv.org/pdf/1511.07528.pdf). The supported
        frameworks for the JSMA include ART. Notably, the JSMA implementation
        in ART assumes the l0-norm is passed in as a percentage (which is why
        we pass in linf).

        :return: JSMA adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        if "art" in self.available:
            from art.attacks.evasion import SaliencyMapMethod

            print("Producing DF adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    SaliencyMapMethod(
                        classifier=self.art_classifier,
                        theta=self.attack_params["alpha"] + 1,
                        gamma=self.linf,
                        batch_size=self.x.size(0),
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["jsma"], tuple(fw for fw in (art_adv,) if fw is not None)

    def pgd(self):
        """
        This method crafts adversarial examples with PGD (Projected Gradient
        Descent)) (https://arxiv.org/pdf/1706.06083.pdf). The supported
        frameworks for PGD include CleverHans and ART.

        :return: PGD adversarial examples
        :rtype: tuple of torch Tensor objects (n, m)
        """
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.projected_gradient_descent import (
                projected_gradient_descent,
            )

            print("Producing PGD adversarial examples with CleverHans...")
            ch_adv = (
                projected_gradient_descent(
                    model_fn=self.attack_params["model"],
                    x=self.x.clone(),
                    eps=self.linf,
                    eps_iter=self.attack_params["alpha"],
                    nb_iter=self.attack_params["epochs"],
                    norm=float("inf"),
                    clip_min=self.attack_params["clip"][0].max().item(),
                    clip_max=self.attack_params["clip"][1].min().item(),
                    y=self.y,
                    targeted=False,
                    rand_init=True,
                    rand_minmax=self.linf,
                    sanity_checks=True,
                ).detach(),
                "CleverHans",
            )
        if "art" in self.available:
            from art.attacks.evasion import ProjectedGradientDescent

            print("Producing PGD adversarial examples with ART...")
            art_adv = (
                torch.from_numpy(
                    ProjectedGradientDescent(
                        estimator=self.art_classifier,
                        norm="inf",
                        eps=self.linf,
                        eps_step=self.attack_params["alpha"],
                        decay=None,
                        max_iter=self.attack_params["epochs"],
                        targeted=False,
                        num_random_init=1,
                        batch_size=self.x.size(0),
                        random_eps=True,
                        summary_writer=False,
                        verbose=True,
                    ).generate(x=self.x.clone().numpy())
                ),
                "ART",
            )
        return self.attacks["pgd"], tuple(
            fw for fw in (ch_adv, art_adv) if fw is not None
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

        :param attack: attack to test
        :type attack: aml Attack object
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
            advx_acc = self.model.accuracy(self.x + p, self.y).item()
            print(f"{attack.name} complete! Model Acc: {advx_acc:.2%},", norm_results)
            self.assertLess(advx_acc, self.min_acc)
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
    https://pytorch.org/docs/stable/testing.html.

    The following frameworks are supported:
        CleverHans (https://github.com/cleverhans-lab/cleverhans)
        ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)

    The following attacks are supported:
        APGD-CE (Auto-PGD with CE loss) (https://arxiv.org/pdf/2003.01690.pdf)
        APGD-DLR (Auto-PGD with DLR loss) (https://arxiv.org/pdf/2003.01690.pdf)
        BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
        CW-L2 (Carlini-Wagner with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf)
        DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
        FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
        JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)
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
                f"{attack.name} AML v. {f}: Min: {mn:.3}, Max {mx:.3},",
                f"Mean: {me:.3}, Median: {md:.3} SD: {st:.3}",
            )
        rtol, atol = torch.testing._comparison.default_tolerances(*diffs)
        for fw, diff in zip(fws, diffs):
            with self.subTest(Attack=f"{attack.name} v. {fw}"):
                close = diff.isclose(torch.zeros_like(diff), atol, rtol).all(1)
                print(
                    f"{close.sum().item()}/{close.numel()}",
                    f"({close.sum().div(close.numel()).item():.2%})",
                    f"of {attack.name} aml adversarial examples are close to {fw}.",
                )

                # if the identity test fails, aim to debug why
                try:
                    torch.testing.assert_close(diff, torch.zeros_like(diff))
                except AssertionError:
                    targets = close == self.model.correct
                    print(
                        f"{attack.name} v {fw} Test failed. Post-analysis:\n",
                        f" - {targets.sum().div(targets.numel()):.2%}",
                        "of failed inputs were initially misclassified",
                    )
                    raise
        return None

    def test_apgdce(self):
        """
        This method performs an identity test for APGD-CE (Auto-PGD with CE
        loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks
        for APGD-CE include ART.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.apgdce())

    def test_apgddlr(self):
        """
        This method performs an identity test for APGD-DLR (Auto-PGD with DLR
        loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks
        for APGD-DLR include ART. Notably, DLR loss is undefined when there are
        only two classes.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.apgddlr())

    def test_bim(self):
        """
        This method performs an identity test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for
        BIM include CleverHans and ART. Notably, the implementation of BIM in
        ART ostensibly always fails, given that ART always uses the current
        model logits in untargetted attacks and thus, *corrects* misclassified
        inputs such that they are classified correctly. In effect, the
        difference between ART and aml adversarial examples from BIM is
        commonly the linf threat model, epsilon.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.bim())

    def test_cwl2(self):
        """
        This method performs an identity test for CW-L2 (Carlini-Wagner with l₂
        norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported frameworks
        for CW-L2 include CleverHans and ART.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.cwl2())

    def test_df(self):
        """
        This method performs an identity test for DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
        include ART.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.df())

    def test_fab(self):
        """
        This method performs an identity test for FAB (Fast Adaptive Boundary)
        (https://arxiv.org/pdf/1907.02044.pdf). The supported frameworks for DF
        include ART.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.fab())

    def test_jsma(self):
        """
        This method performs an identity test for JSMA (Jacobian Saliency Map
        Approach) (https://arxiv.org/pdf/1511.07528.pdf). The supported
        frameworks for the JSMA include ART.

        :return: None
        :rtype: NoneType
        """
        return self.identity_test(*self.jsma())

    def test_pgd(self):
        """
        This method performs an identity test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf).

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
    normalized utilization of lp-norm budget.

    The following frameworks are supported:
        CleverHans (https://github.com/cleverhans-lab/cleverhans)
        ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)

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
        considered to be "semantically" correct. Performance is measured as the
        one minus the l2-norm of normalized decrease in model accuracy and
        normalized utilization of lp-norm budget. Notably, if aml attacks are
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
        if "art" in cls.available:
            from art.estimators.classification import PyTorchClassifier

            cls.art_classifier = PyTorchClassifier(
                model=cls.attack_params["model"].model,
                clip_values=(
                    cls.attack_params["clip"][0].max().item(),
                    cls.attack_params["clip"][1].min().item(),
                ),
                loss=cls.attack_params["model"].loss,
                optimizer=cls.attack_params["model"].optimizer,
                input_shape=cls.x.shape,
                nb_classes=cls.attack_params["model"].params["classes"],
            )
        return None

    def semantic_test(self, attack, fws):
        """
        This method performs a semantic test for a given attack.

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

        # compute perturbation norms
        ps = (aml_p, *fws_p)
        lp = (0, 2, torch.inf)
        norms = tuple([p.norm(d, 1).mean().item() for d in lp] for p in ps)

        # compute model accuracy decrease
        advs = (self.x + aml_p, *fws_adv)
        acc_abs = tuple(self.model.accuracy(adv, self.y).item() for adv in advs)
        acc_dec = tuple(a / self.clean_acc for a in acc_abs)

        # compute perturbation budget consumption
        max_b = (self.l0, self.l2, self.linf)
        atk_b = ((n / b for n, b in zip(an, max_b)) for an in norms)

        # compute performance and organize printing results
        perfs = tuple(
            [max(1 - (aa**2 + b**2) ** (1 / 2), 0) for b in ab]
            for ab, aa in zip(atk_b, acc_dec)
        )
        norms_perfs = (
            ", ".join(
                f"l{p}: {n:.3}/{float(b):.3} ({n/b:.2%}) {f:.1%}"
                for p, n, b, f in zip((0, 2, "∞"), an, max_b, perf)
            )
            for an, perf in zip(norms, perfs)
        )
        acc_diff = (a - self.clean_acc for a in acc_abs)
        results = zip(("AML", *fws), acc_abs, acc_diff, norms_perfs)
        for f, a, d, n in results:
            print(f"{f} {attack.name} Model Acc: {a:.2%} ({d:.2%}), Results: {n}")

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
        loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks
        for APGD-CE include ART.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.apgdce())

    def test_apgddlr(self):
        """
        This method performs a semantic test for APGD-DLR (Auto-PGD with DLR
        loss) (https://arxiv.org/pdf/2003.01690.pdf). The supported frameworks
        for APGD-DLR include ART. Notably, DLR loss is undefined when there are
        only two classes.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.apgddlr())

    def test_bim(self):
        """
        This method performs a semantic test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for
        BIM include CleverHans and ART.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.bim())

    def test_cwl2(self):
        """
        This method performs a semantic test for CW-L2 (Carlini-Wagner with l₂
        norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported frameworks
        for CW-L2 include CleverHans and ART.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.cwl2())

    def test_df(self):
        """
        This method performs a semantic test for DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
        include ART.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.df())

    def test_fab(self):
        """
        This method performs a semantic test for FAB (Fast Adaptive Boundary)
        (https://arxiv.org/pdf/1907.02044.pdf). The supported frameworks for DF
        include ART.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.fab())

    def test_jsma(self):
        """
        This method performs a semantic test for JSMA (Jacobian Saliency Map
        Approach) (https://arxiv.org/pdf/1511.07528.pdf). The supported
        frameworks for the JSMA include ART.

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.jsma())

    def test_pgd(self):
        """
        This method performs a semantic test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.semantic_test(*self.pgd())


class SpecialTest(unittest.TestCase):
    """
    The following class implements the special test case. Special tests are
    designed to exercise components of the Space of Adversarial Strategies
    framework (https://arxiv.org/pdf/2209.04521.pdf) in specific ways. The
    purpose and detailed description of each test can be found in their
    respecitve methods.

    :func:`test_all_components`: coverage test that stresses all components
    """


if __name__ == "__main__":
    raise SystemExit(0)
