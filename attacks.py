"""
This module defines the attacks proposed in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Mon Apr 18 2022
"""
import itertools  # Functions creating iterators for efficietn looping
import loss  # PyTorch-based custom loss functions
import optimizers  # PyTorch-based custom optimizers
import saliency  # Gradient manipulation heuristics to achieve adversarial goals
import surface  # PyTorch-based models for crafting adversarial examples
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
import traveler  # PyTorch-based optimizers for crafting adversarial examples
from utilities import print  # Use timestamped print

# TODO
# implement unit test
# add shortcuts for auto-building known attacks (assist architectures.py)
# check if tuple passed as clip needs to be cast as matrix
# add early termination support
# organize argument orders


class Attack:
    """
    The Attack class serves as a binder between Traveler and Surface objects
    with a high-level interface. Detailed in [paper_url], attacks are built
    from a differentiable function (i.e., a surface) and routines to manipulate
    inputs (i.e., a traveler). Upon instantiation, the `craft` method serves as
    the main entry point in crafting adversarial examples.

    :func:`__init__`: instantiates Attack objects
    :func:`__repr__`: returns the attack name (based on the components)
    :func:`craft`: returns a batch of adversarial examples
    """

    def __init__(
        self,
        epochs,
        clip,
        epsilon,
        alpha,
        change_of_variables,
        optimizer,
        random_restart,
        loss,
        norm,
        model,
        saliency_map,
        batch_size=-1,
        surface_closure=(),
        traveler_closure=(),
    ):
        """
        This method instantiates an Attack object with a variety of parameters
        necessary for building and coupling Traveler and Surface objects. The
        following parameters define high-level bookkeeping parameters across
        attacks:

        :param batch_size: crafting batch size (-1 for 1 batch)
        :type batch_size: int
        :param epochs: number of optimization steps to perform
        :type epochs: int
        :param clip: range of allowable values for the domain
        :type clip: tuple of integers or PyTorch FloatTensor objects (samples, features)
        :param epsilon: lp-norm ball threat model
        :type epsilon: float

        These subsequent parameters define the components of a Traveler object:

        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: bool
        :param optimizer: optimization algorithm to use
        :type optimizer: optimizer module object
        :param random_restart: whether to randomly perturb inputs
        :type random_restart: bool
        :param traveler_closure: subroutines after each perturbation
        :type traveler_closure: tuple of callables

        Finally, the following parameters define Surface objects:

        :param loss: objective function to differentiate
        :type loss: loss module object
        :param norm: lp-space to project gradients into
        :type norm: surface module callable
        :param model: neural network
        :type model: PyTorch Module-inherited object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: saliency module object
        :param surface_closure: subroutines after each backward pass
        :type surface_closure: tuple of callables

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
            "loss function": loss.__name__,
            "optimizer": optimizer.__name__,
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
            set("S", "r̶", "v̶", "Cr", "i", "∞"): "BIM",
            set("A", "r̶", "V", "CW", "i", "2"): "CWL2",
            set("S", "r̶", "v̶", "Id", "d", "2"): "DF",
            set("S", "R", "v̶", "Cr", "i", "∞"): "PGD",
            set("S", "r̶", "v̶", "Id", "j", "0"): "JSMA",
        }
        self.name = name_map.get(name, "-".join(name))
        self.params = {"α": alpha, "ε": epsilon, "epochs": epochs}

        # instantiate traveler, surface, and necessary subcomponents
        loss = loss()
        optimizer = optimizer(
            lr=alpha,
            model_acc=model.accuracy if optimizer.req_acc else None,
            atk_loss=loss if optimizer.req_loss else None,
        )
        self.traveler = traveler.Traveler(
            change_of_variables, optimizer, random_restart, traveler_closure
        )
        self.surface = surface.Surface(model, saliency, loss, norm, surface_closure)
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of attack
        components and parameters. Notably, if the collection of components
        defines an attack made popular in the literature, it's full name is
        returned instead. The following named attacks are supported:

            BIM (Basic Iterative Method) (https://arxiv.org/abs/1607.02533)
            CWL2 (Carlini-Wagner L2) (https://arxiv.org/abs/1608.04644)
            DF (DeepFool) (https://arxiv.org/abs/1511.04599)
            PGD (Projected Gradient Descent) (https://arxiv.org/abs/1706.06083)
            JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/abs/1511.07528)

        :return: the attack name with parameters
        :rtype: str
        """
        return f"{self.name}({self.params})"

    def craft(self, x, y):
        """
        This method crafts adversarial examples, as defined by the attack
        parameters and the instantiated Travler and Surface attribute objects.
        Specifically, it creates a copy of x, creates the desired batch size,
        performs traveler initilizations (i.e., change of variables and random
        restart), and iterates an epoch number of times.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: PyTorch FloatTensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: PyTorch Tensor object (n,)
        :return: a batch of adversarial examples
        :rtype: PyTorch FloatTensor object (n, m)
        """
        x = x.detatch().clone()
        p = x.new_zeros(x.size())
        chunks = len(x) if self.batch_size == -1 else -(-len(x) // self.batch_size)
        for b, (xb, yb, pb) in enumerate(
            zip(x.chunk(chunks), y.chunk(chunks), p.chunk(chunks)), start=1
        ):
            print(f"Crafting {len(x)} adversarial examples with {self}... {b/chunks}")
            min_p, max_p = (
                (clip - xb).clamp(-self.epsilon, self.epsilon) for clip in self.clip
            )
            self.surface.initialize(xb, pb, (min_p, max_p))
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
    epochs=None,
    epsilon=None,
    model=None,
    change_of_variables_enabled=(False, True),
    optimizers=(optimizers.SGD, optimizers.Adam),
    random_restart_enabled=(False, True),
    losses=(loss.IdentityLoss, loss.CrossEntropyLoss, loss.CWLoss),
    norms=(surface.l0, surface.l2, surface.linf),
    saliency_maps=(saliency.identity, saliency.jsma, saliency.deepfool),
):
    """
    As shown in [paper_url], seminal attacks in machine learning can be cast
    into a single, unified framework. With this observation, this method
    combinatorically builds attack objects by swapping popular optimizers,
    norms, saliency maps, loss functions, and other "tricks" used within the
    AML community, such as random restart and change of variables. The
    combinations of supported components are shown below:

        Traveler Components:
        Change of variables: Enabled or disabled
        Optimizer: Adam and Stochastic Gradient Descent
        Random restart: Enabled or disabled

        Surface Components:
        Loss: Identity, Categorical Cross-Entropy, and Carlini-Wagner
        Norm: l₀, l₂, and l∞
        Saliency map: Identity, JSMA, and DeepFool

    Moreover, we expose these support components as arguments above for ease of
    instantiating a subset of the total combination space.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: PyTorch Module-inherited object
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
        optimizer,
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
            optimizer=optimizer,
            alpha=alpha,
            random_restart=random_restart,
            change_of_variables=change_of_variables,
            model=model,
            saliency_map=saliency_map,
            loss=loss_func,
            norm=norm,
        )


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
