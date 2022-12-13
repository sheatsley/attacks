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


class BaseTest(unittest.TestCase):
    """
    The following class serves as the base test for all tests cases. Since
    there is no good way to setup test fixtures *across test cases*, we emulate
    this setup by defining a base class which has a class setup test fixture
    (Unfortunately, setUpModule does nothing special with namespaces, so it is
    ostensibly useless even though it seems like it ought to fit the bill...).

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
            (mins[0], maxs[0])
            if (mins[0].eq(mins).all() and maxs[0].eq(maxs).all())
            else f"(({mins.min().item()}, ..., {mins.max()}.item()),"
            "({maxs.min().item()}, ..., {maxs.max().item()}))"
        )

        # train model
        template = getattr(dlm.architectures, dataset)
        cls.model = (
            dlm.CNNClassifier(template=template)
            if template.CNNClassifier is not None
            else dlm.MLPClassifier(template=template)
        )
        cls.model.fit(*(x, y) if has_test else (cls.x, cls.y))

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
            "APGD-CE": aml.attacks.apgdce(**cls.attack_params | {"epsilon": cls.linf}),
            "APGD-DLR": aml.attacks.apgdce(**cls.attack_params | {"epsilon": cls.linf}),
            "BIM": aml.attacks.bim(**cls.attack_params | {"epsilon": cls.linf}),
            "CW-L2": aml.attacks.cwl2(**cls.attack_params | {"epsilon": cls.l2}),
            "DF": aml.attacks.df(**cls.attack_params | {"epsilon": cls.l2}),
            "FAB": aml.attacks.fab(**cls.attack_params | {"epsilon": cls.l2}),
            "JSMA": aml.attacks.fab(**cls.attack_params | {"epsilon": cls.l0}),
            "PGD": aml.attacks.fab(**cls.attack_params | {"epsilon": cls.linf}),
        }

        # determine available frameworks (and import for test cases that need it)
        supported = ("cleverhans", "art")
        cls.available = [f for f in supported if importlib.util.find_spec(f)]
        frameworks = ", ".join(cls.available)
        print(
            "Module Setup complete. Testing Parameters:",
            f"Dataset: {dataset}, Test Set: {has_test}",
            f"Craftset shape: ({cls.x.size(0)}, {cls.x.size(1)})",
            f"Model Type: {cls.model.__class__.__name__}",
            f"Train Acc: {cls.model.stats['train_acc'][-1]:.1%}",
            f"Craftset Acc: {cls.model.accuracy(cls.x, cls.y):.1%}",
            f"Attack Clipping Values: {clip_info}",
            f"Attack Strength α: {cls.attack_params['alpha']}",
            f"Attack Epochs: {cls.attack_params['epochs']}",
            f"Max Norm Radii: l0: {cls.l0}, l2: {cls.l2:.3}, l∞: {cls.linf}",
            f"Available Frameworks: {frameworks}",
            sep="\n",
        )
        return None


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
            advx_acc = self.model.accuracy(self.x + p, self.y)
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
        return self.functional_test(self.attacks["APGD-CE"])

    def test_apgddlr(self):
        """
        This method performs a functional test for APGD-DLR (Auto-PGD with DLR
        loss) (https://arxiv.org/pdf/2003.01690.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["APGD-DLR"])

    def test_bim(self):
        """
        This method performs a functional test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["BIM"])

    def test_cwl2(self):
        """
        This method performs a functional test for CW-L2 (Carlini-Wagner with
        l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["CW-L2"])

    def test_df(self):
        """
        This method performs a functional test for DF (DeepFool)
        (https://arxiv.org/pdf/1511.04599.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["DF"])

    def test_fab(self):
        """
        This method performs a functional test for FAB (Fast Adaptive Boundary)
        (https://arxiv.org/pdf/1907.02044.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["FAB"])

    def test_jsma(self):
        """
        This method performs a functional test for JSMA (Jacobian Saliency Map
        Approach) (https://arxiv.org/pdf/1511.07528.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["JSMA"])

    def test_pgd(self):
        """
        This method performs a functional test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(self.attacks["PGD"])


class IdentityTests(unittest.TestCase):
    """
    The following class implements identity tests. Identity correctness tests
    assert that the feature values of aml adversarial examples themselves must
    be at least 99% similar to adversarial examples of other frameworks.
    Feature values are considered similar if their difference is smaller than ε
    (defaulted to 0.001).

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
        This method initializes the identity testing case by setting parameters
        unique to the tests within this test case. At this time, this sets the
        maximum allowable distance (measured at the feature-level) between
        adversarial examples produced by aml and other frameworks such that aml
        adversarial examples are considered "identical" to adversarial examples
        produced by attacks in other frameworks.

        :param max_distance: the maximum allowable distance between features
        :type max_distance: float
        :return: None
        :rtype: NoneType
        """
        cls.max_distance = max_distance
        return None


class SemanticTests(BaseTest):
    """
    The following class implements semantic tests. Semantic correctness tests
    are more sophisticated than functional tests, in that they compare the the
    performance of adversarial examples produced by known attacks within aml to
    those implemented in other adversarial machine learning frameworks. Attacks
    in aml are determined to be semantically correct if the produced
    adversarial examples are within no worse than 1% of the performance of
    adversarial examples produced by other frameworks. Performance is defined
    as one minus the product of model accuracy and lp-norm (normalized to 0-1).

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
        one minus the product of model accuracy and normalized lp-norm.
        Notably, if aml attacks are *better* than other frameworks by greater
        than the maximum allowable performance degradation, the tests still
        pass (failure can only occurs if aml attacks are worse).

        :param max_perf_degrad: the maximum allowable difference in performance
        :type max_perf_degrad: float
        :return: None
        :rtype: NoneType
        """
        super().setUpClass()
        cls.max_perf_degrad = max_perf_degrad
        return None

    def semantic_test(self, attack, fws):
        """
        This method performs a semantic test for a given attack.

        :param attack: attack to test
        :type attack: aml Attack object
        :param fws: adversarial examples produced by other frameworks
        :type fws: tuple of torch Tensor object (n, m) and str
        :return: None
        :rtype: NoneType
        """
        for fw_adv, fw in fws:
            with self.subTest(Attack=f"{attack.name} v. {fw}"):

                # craft adversarial examples
                aml_p = attack.craft(self.x, self.y)
                fw_p = fw_adv.sub(self.x)

                # compute perturbation norms
                budgets = (self.l0, self.l2, self.linf)
                pert_norms = zip((aml_p, fw_p), (0, 2, torch.inf) * 2)
                aml_norm, fw_norm = (p.norm(d, 1).mean().item() for p, d in pert_norms)

                # compute model accuracy
                advs = (self.x + aml_p, fw_adv)
                aml_acc, fw_acc = (self.model.accuracy(adv, self.y) for adv in advs)

                # compute perturbation budget consumption
                norm_budgets = zip((aml_norm, fw_norm), budgets * 2)
                aml_budget, fw_budget = (n / b for n, b in norm_budgets)
                acc_budgets = zip((aml_acc, fw_acc), (aml_budget, fw_budget))

                # compute performance and organize printing results
                perf = list(zip((1 - a * b for a, b in acc_budgets)))
                perf_norms = (
                    (f"l{p}: {n:.3}/{b} ({n/b:.2%})", f"l{p}: {f:.2%}")
                    for norms, perf in zip((aml_norm, fw_norm), perf)
                    for n, b, p, f in zip(norms, budgets, (0, 2, "∞"), perf)
                )
                results = zip(("AML", fw), (aml_acc, fw_acc), perf_norms)
                print(
                    f"{f} {attack.name} Model Acc: {f:.2%}, "
                    "Norms: {', '.join(n)}, "
                    "Performance: {', '.join(perf)}"
                    for f, a, n in results
                )

                # compute target norm and assert marginal performance difference
                norm_map = (aml.surface.l0, aml.surface.l2, aml.surface.linf)
                aml_perf, fw_perf = perf[norm_map.index(attack.surface.norm)]
                self.assertLessEqual(aml_perf - self.max_perf_degrad, fw_perf)
        return None

    def test_bim(self):
        """
        This method performs a semantic test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks
        for BIM include CleverHans and ART.

        :return: None
        :rtype: NoneType
        """
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.projected_gradient_descent import (
                projected_gradient_descent as basic_iterative_method,
            )

            print("Producing BIM adversarial examples with CleverHans...")
            ch_adv = (
                basic_iterative_method(
                    model_fn=self.attack_params["model"],
                    x=self.x,
                    eps=self.linf,
                    eps_iter=self.attack_params["alpha"],
                    nb_iter=self.attack_params["epochs"],
                    clip_min=self.attack_params["clip"][0].max().item(),
                    clip_max=self.attack_params_["clip"][1].min().item(),
                    y=self.y,
                    targeted=False,
                    rand_init=False,
                    rand_minimax=0,
                    sanity_checks=True,
                ),
                "CleverHans",
            )
        if "art" in self.available:
            from art.attacks.evasion import BasicIterativeMethod
            from art.estimators.classification import PyTorchClassifier

            print("Producing BIM adversarial examples with ART...")
            art_adv = (
                BasicIterativeMethod(
                    classifier=PyTorchClassifier(
                        model=self.attack_params["model"],
                        clip_values=(
                            self.attack_params["clip"][0].max().item(),
                            self.attack_params["clip"][1].min().item(),
                        ),
                        loss=self.attack_params["model"].loss,
                        optimizer=self.attack_params["model"].optimizer,
                        input_shape=self.x.shape,
                        nb_classes=self.attack_params["model"].params["classes"],
                    ),
                    eps=self.linf,
                    eps_step=self.attack_params["alpha"],
                    max_iter=self.attack_params["epochs"],
                    targeted=False,
                    batch_size=self.x.size(0),
                    verbose=True,
                ).generate(x=self.x),
                "ART",
            )
        return self.semantic_test(self.attacks["BIM"], (ch_adv, art_adv))

    def test_cwl2(self):
        """
        This method performs a semantic test for CW-L2 (Carlini-Wagner with l₂
        norm) (https://arxiv.org/pdf/1608.04644.pdf). The supported frameworks
        for CW-L2 include CleverHans and ART.

        :return: None
        :rtype: NoneType
        """
        if "cleverhans" in self.available:
            from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

            print("Producing CW-L2 adversarial examples with CleverHans...")
            ch_adv = (
                carlini_wagner_l2(
                    model_fn=self.attack_params["model"],
                    x=self.x,
                    n_classes=self.attack_params["model"].params["classes"],
                    y=self.y,
                    lr=self.attack_params["alpha"],
                    confidence=self.attacks["CW-L2"].surface.loss.k,
                    clip_min=self.attack_params["clip"][0].max().item(),
                    clip_max=self.attack_params_["clip"][1].min().item(),
                    initial_const=self.attacks["CW-L2"].surface.loss.c.item(),
                    binary_search_steps=1,
                    max_iterations=self.attack_params["epochs"],
                ),
                "CleverHans",
            )
        if "art" in self.available:
            from art.attacks.evasion import CarliniL2Method
            from art.estimators.classification import PyTorchClassifier

            print("Producing CW-L2 adversarial examples with ART...")
            art_adv = (
                CarliniL2Method(
                    classifier=PyTorchClassifier(
                        model=self.attack_params["model"],
                        clip_values=(
                            self.attack_params["clip"][0].max().item(),
                            self.attack_params["clip"][1].min().item(),
                        ),
                        loss=self.attack_params["model"].loss,
                        optimizer=self.attack_params["model"].optimizer,
                        input_shape=self.x.shape,
                        nb_classes=self.attack_params["model"].params["classes"],
                    ),
                    confidence=self.attacks["CW-L2"].surface.loss.k,
                    targeted=False,
                    learning_rate=self.attack_params["alpha"],
                    binary_search_steps=1,
                    max_iter=self.attack_params["epochs"],
                    initial_const=self.attacks["CW-L2"].surface.loss.c.item(),
                    max_halving=5,
                    max_doubling=5,
                    batch_size=self.x.size(0),
                    verbose=True,
                ).generate(x=self.x),
                "ART",
            )
        return self.semantic_test(self.attacks["CW-L2"], (ch_adv, art_adv))

    def test_df(self):
        """
        This method performs a semantic test for DF (DeepFool)
        (https://arxiv.org/pdf/1611.01236.pdf). The supported frameworks for DF
        include ART.

        :return: None
        :rtype: NoneType
        """
        if "art" in self.available:
            from art.attacks.evasion import DeepFool
            from art.estimators.classification import PyTorchClassifier

            print("Producing DF adversarial examples with ART...")
            art_adv = (
                DeepFool(
                    classifier=PyTorchClassifier(
                        model=self.attack_params["model"],
                        clip_values=(
                            self.attack_params["clip"][0].max().item(),
                            self.attack_params["clip"][1].min().item(),
                        ),
                        loss=self.attack_params["model"].loss,
                        optimizer=self.attack_params["model"].optimizer,
                        input_shape=self.x.shape,
                        nb_classes=self.attack_params["model"].params["classes"],
                    ),
                    max_iter=self.attack_params["epochs"],
                    epsilon=0,
                    nb_grads=self.attack_params["model"].params["classes"],
                    batch_size=self.x.size(0),
                    verbose=True,
                ).generate(x=self.x),
                "ART",
            )
        return self.semantic_test(self.attacks["DF"], (art_adv,))


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
