"""
This module defines the attacks proposed in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Mon Apr 18 2022
"""
import itertools  # Functions creating iterators for efficietn looping
import loss  # PyTorch-based custom loss functions
import optimizer  # PyTorch-based custom optimizers
import saliency  # Gradient manipulation heuristics to achieve adversarial goals
import surface  # PyTorch-based models for crafting adversarial examples
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
import traveler  # PyTorch-based optimizers for crafting adversarial examples
from utilities import print  # Use timestamped print

# TODO
# implement unit test
# add early termination support
# implement DDN


class Attack:
    """
    The Attack class serves as a binder between Traveler and Surface objects
    with a high-level interface. Detailed in
    https://arxiv.org/pdf/2209.04521.pdf, attacks are built from a
    differentiable function (i.e., a surface) and routines to manipulate inputs
    (i.e., a traveler). Upon instantiation, the `craft` method serves as the
    main entry point in crafting adversarial examples.

    :func:`__init__`: instantiates Attack objects
    :func:`__repr__`: returns the attack name (based on the components)
    :func:`craft`: returns a batch of adversarial examples
    """

    def __init__(
        self,
        clip,
        epochs,
        epsilon,
        alpha,
        change_of_variables,
        optimizer_alg,
        random_restart,
        loss_func,
        norm,
        model,
        saliency_map,
        batch_size=-1,
    ):
        """
        This method instantiates an Attack object with a variety of parameters
        necessary for building and coupling Traveler and Surface objects. The
        following parameters define high-level bookkeeping parameters across
        attacks:

        :param batch_size: crafting batch size (-1 for 1 batch)
        :type batch_size: int
        :param clip: range of allowable values for the domain
        :type clip: tuple of floats or torch Tensor object (n, m)
        :param epochs: number of optimization steps to perform
        :type epochs: int
        :param epsilon: lp-norm ball threat model
        :type epsilon: float

        These subsequent parameters define the components of a Traveler object:

        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: bool
        :param optimizer_alg: optimization algorithm to use
        :type optimizer_alg: optimizer module class
        :param random_restart: whether to randomly perturb inputs
        :type random_restart: bool

        Finally, the following parameters define Surface objects:

        :param loss_func: objective function to differentiate
        :type loss_func: loss module class
        :param norm: lp-space to project gradients into
        :type norm: surface module callable
        :param model: neural network
        :type model: scikit-torch LinearClassifier-inherited object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: saliency module class

        To easily identify attacks, __repr__ is overriden and instead returns
        an abbreviation computed by concatenating the first letter (or two, if
        there is a name collision) of the following parameters, in order: (1)
        optimizer, (2) if random restart was used, (3) if change of variables
        was applied, (4) loss function, (5) saliency map, and (6) norm used.
        Combinations that yield known attacks are labeled as such (see __repr__
        for more details).

        :return: an attack
        :rtype: Attack object
        """

        # save attack parameters and build short attack name
        self.batch_size = batch_size
        self.epochs = epochs
        self.epsilon = epsilon
        self.clip = clip
        self.components = {
            "change of variables": change_of_variables,
            "loss function": loss_func.__name__,
            "optimizer": optimizer_alg.__name__,
            "random restart": random_restart,
            "saliency map": saliency_map.__name__,
            "target norm": "l∞" if norm == surface.linf else norm.__name__,
        }
        name = {
            self.components["optimizer"][0],
            "R" if self.components["random restart"] else "r̶",
            "V" if self.components["change of variables"] else "v̶",
            self.components["loss function"][:2],
            self.components["saliency map"][0],
            self.components["target norm"][1],
        }
        name_map = {
            set("M", "R", "v̶", "CE", "i", "∞"): "APGD-CE",
            set("M", "R", "v̶", "DL", "i", "∞"): "APGD-DLR",
            set("S", "r̶", "v̶", "CE", "i", "∞"): "BIM",
            set("A", "r̶", "V", "CW", "i", "2"): "CW-L2",
            set("S", "r̶", "v̶", "Id", "d", "2"): "DF",
            set("B", "r̶", "v̶", "Id", "d", "2"): "FAB",
            set("S", "R", "v̶", "CE", "i", "∞"): "PGD",
            set("S", "r̶", "v̶", "Id", "j", "0"): "JSMA",
        }
        self.name = name_map.get(name, "-".join(name))
        self.params = {"α": alpha, "ε": epsilon, "epochs": epochs}

        # instantiate traveler, surface, and necessary subcomponents
        norm_map = {0: surface.l0, 1: surface.linf, 2: surface.l2}
        saliency_map = (
            saliency_map(norm_map[norm])
            if saliency_map is saliency.DeepFoolSaliency
            else saliency_map()
        )
        loss_func = loss_func()
        custom_opt_params = {
            "atk_loss": loss_func,
            "epochs": epochs,
            "epsilon": self.epsilon,
            "model_acc": model,
        }
        torch_opt_params = {
            "lr": self.alpha,
            "maximize": loss_func.max_obj,
            "params": torch.zeros(1),
        }
        optimizer_alg = optimizer_alg(
            **(custom_opt_params | torch_opt_params)
            if optimizer_alg.__module__ == "optimizer"
            else torch_opt_params
        )
        self.traveler = traveler.Traveler(
            change_of_variables, optimizer_alg, random_restart
        )
        self.surface = surface.Surface(model, saliency_map, loss_func, norm)
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of attack
        components and parameters. Notably, if the collection of components
        defines an attack made popular in the literature, it's full name is
        returned instead. The following named attacks are supported:

            APGD-CE (Auto-PGD with CE loss) (https://arxiv.org/pdf/2003.01690.pdf)
            APGD-DLR (Auto-PGD with DLR loss) (https://arxiv.org/pdf/2003.01690.pdf)
            BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
            CW-L2 (Carlini-Wagner with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf)
            DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
            FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
            PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)
            JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)

        :return: the attack name with parameters
        :rtype: str
        """
        return f"{self.name}({self.params})"

    def craft(self, x, y):
        """
        This method crafts adversarial examples, as defined by the attack
        parameters and the instantiated Travler and Surface attribute objects.
        Specifically, it normalizes inputs to be within [0, 1] (as some
        techniques assume this range, e.g., change of variables), creates a
        copy of x, creates the desired batch size, performs traveler
        initilizations (i.e., change of variables and random restart), and
        iterates an epoch number of times.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: a batch of adversarial examples
        :rtype: torch Tensor object (n, m)
        """

        # clone inputs, setup perturbation vector, chunks, epsilon, and clip
        x = x.detatch().clone()
        p = x.new_zeros(x.size())
        chunks = len(x) if self.batch_size == -1 else -(-len(x) // self.batch_size)
        epsilon = (-self.epsilon, self.epsilon)
        clip = (
            self.clip
            if isinstance(self.clip, torch.Tensor)
            else torch.tensor(self.clip).tile(*x.size(), 1)
        )

        # craft adversarial examples per batch
        for b, (xb, yb, pb) in enumerate(
            zip(x.chunk(chunks), y.chunk(chunks), p.chunk(chunks)), start=1
        ):
            print(f"Crafting {len(x)} adversarial examples with {self}... {b/chunks}")
            min_p, max_p = clip.sub(xb.unsqueeze(1)).clamp(*epsilon).unbind(2)
            self.surface.initialize((min_p, max_p))
            self.traveler.initialize(xb, pb)
            p.clamp_(min_p, max_p)
            for epoch in range(self.epochs):
                print(f"On epoch {epoch}... ({epoch/self.epochs:.1%})")
                self.surface(xb, yb, pb)
                self.traveler()
                p.clamp_(min_p, max_p)
        return x + p


def attack_builder(
    alpha=None,
    clip=None,
    epochs=None,
    epsilon=None,
    model=None,
    change_of_variables_enabled=(True, False),
    optimizers=(
        optimizer.Adam,
        optimizer.BackwardSGD,
        optimizer.MomentumBestStart,
        optimizer.SGD,
    ),
    random_restart_enabled=(True, False),
    losses=(loss.CELoss, loss.CWLoss, loss.DLRLoss, loss.IdentityLoss),
    norms=(surface.l0, surface.l2, surface.linf),
    saliency_maps=(
        saliency.DeepFoolSaliency,
        saliency.JacobianSaliency,
        saliency.IdentitySaliency,
    ),
):
    """
    As shown in https://arxiv.org/pdf/2209.04521.pdf, seminal attacks in
    machine learning can be cast into a single, unified framework. With this
    observation, this method combinatorically builds attack objects by swapping
    popular optimizers, norms, saliency maps, loss functions, and other
    techniques used within the AML community, such as random restart and change
    of variables. The combinations of supported components are shown below:

        Traveler Components:
        Change of variables: Enabled or disabled
        Optimizer: Adam, Backward Stochastic Gradient Descent,
                    Momentum Best Start, and Stochastic Gradient Descent
        Random restart: Enabled or disabled

        Surface Components:
        Loss: Categorical Cross-Entropy, Difference of Logits Ratio, Identity,
                and Carlini-Wagner
        Norm: l₀, l₂, and l∞
        Saliency map: DeepFool, Jacobian, and Identity

    We expose these supported components as arguments above for ease of
    instantiating a subset of the total combination space. Moreover, we also
    expose the following experimental parameters below to compare attacks.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: permissible values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: a generator yielding attack combinations
    :rtype: generator of Attack objects
    """

    # generate combinations of components and instantiate Attack objects
    num_attacks = (
        len(optimizers)
        * len(random_restart_enabled)
        * len(change_of_variables_enabled)
        * len(saliency_maps)
        * len(norms)
        * len(losses)
    )
    print(f"Yielding {num_attacks} attacks...")
    for (
        optimizer_alg,
        random_restart,
        change_of_variables,
        loss_func,
        saliency_map,
        norm,
        jacobian,
    ) in itertools.product(
        optimizers,
        random_restart_enabled,
        change_of_variables_enabled,
        losses,
        saliency_maps,
        norms,
    ):
        yield Attack(
            epochs=epochs,
            optimizer=optimizer_alg,
            alpha=alpha,
            random_restart=random_restart,
            change_of_variables=change_of_variables,
            model=model,
            saliency_map=saliency_map,
            loss=loss_func,
            norm=norm,
        )


def apgdce(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build Auto-PGD with Cross-Entropy loss
    (APGD-CE), as shown in https://arxiv.org/abs/2003.01690. Specifically,
    APGD-CE: does not use change of variables, uses the Momentum Best Start
    optimizer, uses random restart, uses Cross-Entropy loss, uses l∞ norm, and
    uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: APGD-CE attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.MomentumBestStart,
        random_restart=True,
        loss=loss.CELoss,
        norm=surface.linf,
        saliency_map=saliency.IdentitySaliency,
    )


def apgddlr(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build Auto-PGD with the Difference of
    Logits Ratio loss (APGD-DLR), as shown in
    https://arxiv.org/abs/2003.01690. Specifically, APGD-DLR: does not use
    change of variables, uses the Momentum Best Start optimizer, uses random
    restart, uses Difference of Logits ratio loss, uses l∞ norm, and uses the
    Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: APGD-DLR attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.MomentumBestStart,
        random_restart=True,
        loss=loss.DLRLoss,
        norm=surface.linf,
        saliency_map=saliency.IdentitySaliency,
    )


def bim(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build the Basic Iterative Method (BIM),
    as shown in (https://arxiv.org/pdf/1611.01236.pdf) Specifically, BIM: does
    not use change of variables, uses the Stochastic Gradient Descent
    optimizer, does not use random restart, uses Cross Entropyy loss, uses l∞
    norm, and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: BIM attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.SGD,
        random_restart=False,
        loss=loss.CELoss,
        norm=surface.linf,
        saliency_map=saliency.IdentitySaliency,
    )


def cwl2(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build Carlini-Wagner l₂ (CW-L2), as
    shown in (https://arxiv.org/pdf/1608.04644.pdf) Specifically, CW-L2: uses
    change of variables, uses the Adam optimizer, does not use random restart,
    uses Carlini-Wagner loss, uses l₂ norm, and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: Carlini-Wagner l₂ attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=True,
        optimizer=optimizer.Adam,
        random_restart=False,
        loss=loss.CWLoss,
        norm=surface.l2,
        saliency_map=saliency.IdentitySaliency,
    )


def df(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build DeepFool (DF), as shown in
    (https://arxiv.org/pdf/1511.04599.pdf) Specifically, DF: does not use
    change of variables, uses the Stochastic Gradient Descent optimizer, does
    not use random restart, uses Identity loss, uses l₂ norm, and uses the
    DeepFool saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.SGD,
        random_restart=False,
        loss=loss.IdentityLoss,
        norm=surface.l2,
        saliency_map=saliency.DeepFoolSaliency,
    )


def fab(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build Fast Adaptive Boundary (FAB), as
    shown in (https://arxiv.org/pdf/1907.02044.pdf) Specifically, FAB: does not
    use change of variables, uses the Backward Stochastic Gradient Descent
    optimizer, does not use random restart, uses Identity loss, uses l₂ norm,
    and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.BackwardSGD,
        random_restart=False,
        loss=loss.IdentityLoss,
        norm=surface.l2,
        saliency_map=saliency.IdentitySaliency,
    )


def pgd(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build Projected Gradient Descent (PGD),
    as shown in (https://arxiv.org/pdf/1706.06083.pdf) Specifically, PGD: does
    not use change of variables, uses the Stochastic Gradient Descent
    optimizer, uses random restart, uses Identity loss, uses l∞ norm, and uses
    the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.SGD,
        random_restart=False,
        loss=loss.CELoss,
        norm=surface.linf,
        saliency_map=saliency.IdentitySaliency,
    )


def jsma(alpha=None, clip=None, epochs=None, epsilon=None, model=None):
    """
    This function serves as an alias to build the Jacobian-based Saliency Map
    Approach (JSMA), as shown in (https://arxiv.org/pdf/1511.07528.pdf)
    Specifically, the JSMA: does not use change of variables, uses the
    Stochastic Gradient Descent optimizer, does not use random restart, uses
    Identity loss, uses l0 norm, and uses the Jacobian saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer=optimizer.SGD,
        random_restart=False,
        loss=loss.IdentityLoss,
        norm=surface.l0,
        saliency_map=saliency.JacobianSaliency,
    )


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
