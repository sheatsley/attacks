"""
This module runs attack performance and correctness tests against other AML
libraries. Specifically, this defines three types of tests: (1) functional, (2)
semantic, and (3) identity. Details surrounding these tests can be found in the
respecetive classes: FunctionalTest, SemanticTest, and IdentityTest. Aside from
these standard tests, special tests can be found in the SpecialTest class,
which evaluate particulars of the implementation in the Space of Adversarial
Strategies (https://arxiv.org/pdf/2209.04521.pdf).
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
# add perturbation visualizations (maybe provide in examples?)
# running module direcetly should run all test classes a test suite?
# unclear if test classes should be per-dataset, or per-test-type (need to look at command line patterns)
# add unittest.main() at the end of the script to run *all* tests


class FunctionalTests(unittest.TestCase):
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

    :func:`functional_test`: performs functional test(s)
    :func:`setUp`: loads frameworks, retrieves data, and trains models
    :func:`test_all`: perform functional test for all attacks
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
    def setUpClass(
        cls, alpha=0.01, dataset="phishing", debug=False, epochs=30, small=0.15
    ):
        """
        This method initializes the functional testing framework by retrieving
        data, procesing data, training models, defining supported attacks, and
        saving attack parameters for use later. Moreover, it supports loading
        PyTorch in debug mode to assist debugging errors in autograd.

        :param alpha: perturbation strength
        :type alpha: float
        :param dataset: dataset to run tests over
        :type dataset: str
        :param debug: whether to set the autograd engine in debug mode
        :type debugf: bool
        :param epochs: number of attack iterations
        :type epochs: int
        :param small: maximum % of lp-budget consumption
        :type small: float
        :return: None
        :rtype: NoneType
        """

        # set debug mode, load data (extract training and test sets, if they exist)
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
        maximum, idx = cls.x.max(0)
        minimum, idx = cls.x.min(0)
        clip = torch.stack((minimum, maximum), 1).tile((cls.x.size(0), 1, 1))

        # train model
        template = getattr(dlm.architectures, dataset)
        cls.model = (
            dlm.CNNClassifier(template=template)
            if template.CNNClassifier is not None
            else dlm.MLPClassifier(template=template)
        )
        cls.model.fit(*(x, y) if has_test else (cls.x, cls.y))

        # define supported attacks and save attack parameters
        cls.l0 = int(cls.x.size(1) * small)
        cls.l2 = maximum.sub(minimum).norm(2).item()
        cls.linf = small
        cls.attack_parameters = {
            "alpha": alpha,
            "clip": clip,
            "epochs": epochs,
            "model": cls.model,
        }
        cls.all_attacks = (
            cls.test_apgdce,
            cls.test_apgddlr,
            cls.test_bim,
            cls.test_cwl2,
            cls.test_df,
            cls.test_fab,
            cls.test_jsma,
            cls.test_pgd,
        )
        print(
            f"Setup complete. Dataset: {dataset},",
            f"Craftset shape: ({cls.x.size(0)}, {cls.x.size(1)}),",
            f"Model: {cls.model.__class__.__name__},",
            f"Train Acc: {cls.model.stats['train_acc'][-1]:.1%},",
            f"Max Norm Radii: l0: {cls.l0}, l2: {cls.l2:.3}, l∞: {cls.linf}",
        )
        return None

    def functional_test(self, attack):
        """
        This method performs a functional test for a given attack.

        :param attack: attacks to test
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
            self.assertLess(advx_acc, 0.01)
        return None

    def test_all(self):
        """
        This method performs a functional test for each supported attack.

        :return: None
        :rtype: Nonetype
        """
        return [self.functional_test(attack()) for attack in self.all_attacks]

    def test_apgdce(self):
        """
        This method performs a functional test for APGD-CE (Auto-PGD with CE
        loss) (https://arxiv.org/pdf/2003.01690.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.apgdce(**self.attack_parameters | {"epsilon": self.linf})
        )

    def test_apgddlr(self):
        """
        This method performs a functional test for APGD-DLR (Auto-PGD with DLR
        loss) (https://arxiv.org/pdf/2003.01690.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.apgddlr(**self.attack_parameters | {"epsilon": self.linf})
        )

    def test_bim(self):
        """
        This method performs a functional test for BIM (Basic Iterative Method)
        (https://arxiv.org/pdf/1611.01236.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.bim(**self.attack_parameters | {"epsilon": self.linf})
        )

    def test_cwl2(self):
        """
        This method performs a functional test for CW-L2 (Carlini-Wagner with
        l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.cwl2(**self.attack_parameters | {"epsilon": self.l2})
        )

    def test_df(self):
        """
        This method performs a functional test for DF (DeepFool)
        (https://arxiv.org/pdf/1511.04599.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.df(**self.attack_parameters | {"epsilon": self.l2})
        )

    def test_fab(self):
        """
        This method performs a functional test for FAB (Fast Adaptive Boundary)
        (https://arxiv.org/pdf/1907.02044.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.fab(**self.attack_parameters | {"epsilon": self.l2})
        )

    def test_jsma(self):
        """
        This method performs a functional test for JSMA (Jacobian Saliency Map
        Approach) (https://arxiv.org/pdf/1511.07528.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.jsma(**self.attack_parameters | {"epsilon": self.l0})
        )

    def test_pgd(self):
        """
        This method performs a functional test for PGD (Projected Gradient
        Descent) (https://arxiv.org/pdf/1706.06083.pdf).

        :return: None
        :rtype: NoneType
        """
        return self.functional_test(
            aml.attacks.pgd(**self.attack_parameters | {"epsilon": self.linf})
        )


"""
(1) Functional correctness tests involve crafting
adversarial examples and verifying that model accuracy can be dropped to <1%
for some "small" lp-budget, (2) Semantic correctness tests are more
sophisticated, in that they compare the adversarial examples produced by known
attacks within aml to those implemented in other adversarial machine learning
frameworks. Attacks in aml are determined to be semantically correct if the
produced adversarial examples are within no worse than 1% of the performance of
adversarial examples produced by other frameworks. Performance is defined as
one minus the product of model accuracy and lp-norm (normalized to 0-1). (3)
Identity correctness tests assert that the feature values of aml adversarial
examples themselves must be at least 99% similar to adversarial examples of
other frameworks. Feature values are considered similar if their difference is
smaller than ε (defaulted to 0.001).

    The following frameworks are supported:
        CleverHans (https://github.com/cleverhans-lab/cleverhans)
        ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
"""
"""
    Aside from standard tests, the following unique tests are supported:
        (1) Full component test: performs functional tests with attacks that
        uniquely exercise all possible components with the aml framework.

    :func:`test_all_components`: functional test that excersises all components
    :func:`test_all_functional`: functional test for all attacks
    :func:`test_all_identity`: identity test for all attacks
    :func:`test_all_semantic`: semantic test for all attacks

        # check the frameworks that are available
        supported = ("cleverhans", "art")
        available = ((f, importlib.util.find_spec(f)) for f in supported)
        frameworks = ", ".join((f for f, a in available if a))
"""

if __name__ == "__main__":
    raise SystemExit(0)
