"""
This module defines the attacks proposed in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Wed Jul 28 2021
"""
import itertools  # Functions creating iterators for efficietn looping
import loss  # PyTorch-based custom loss functions
import saliency  # Gradient manipulation heuristics to achieve adversarial goals
import surface  # Classes for rapidly building cost surfaces
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
import traveler  # Classes for rapidly building optimizers
from utilities import print  # Use timestamped print

# TODO
# implement unit test
# add shortcuts for auto-building known attacks (assist architectures.py)
# remove string definitions for saliency maps in auto-builder (use classes/objects)
# let attacks be instantiated without having the model ready (for adversarial training)


class Attack:
    """
    The attack class serves as a binder between Traveler and Surface objects
    with a high-level interface. As shown in [paper_url], an attack is
    instantiated by building a cost function (from which a surface is produced)
    and defining an optimizer (i.e., a traveler) that minimizes the cost
    function. After an attack object is instantiated, its main entry point,
    craft, consumes a tuple of inputs, labels, and parameters, for which it
    returns adversarial examples that are designed to meet a defined
    adversarial objective.

    :func:`__init__`: instantiates Attack objects
    :func:`craft`: returns a batch of adversarial examples
    """

    def __init__(
        self,
        epochs,
        optimizer,
        alpha,
        random_alpha,
        change_of_variables,
        model,
        saliency_map,
        loss,
        norm,
        jacobian,
    ):
        """
        This method instantiates an attack object with a variety of parameters
        necessary for building and coupling Traveler and Surface objects. The
        following parameters define the components of a Traveler object:

        :param epochs: number of optimization steps to perform
        :type epochs: integer
        :param optimizer: optimization algorithm to use
        :type optimizer: PyTorch Optimizer object
        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param random_alpha: magnitude of a random perturbation
        :type random_alpha: scalar
        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: boolean

        While the following parameters define Surface objects:

        :param model: neural network
        :type model: Model object
        :param loss: objective function to differentiate
        :type loss: Loss object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: Saliency object
        :param norm: lp-space to project gradients into
        :type norm: int; one of: 0, 2, or float("inf")
        :param jacobian: source to compute jacobian from
        :type jacobian: string; one of "model" or "bpda"
        :return: a surface
        :rtype: Surface object

        Finally, surfaces depend on Saliency objects, whose parameters are:

        :param saliency_type: the type of saliency map to use
        :type saliency_type: string; one of "identity", "jsma", or "deepfool"

        Thus, the Attack class returns an attack, with a traveler, surface, and
        saliency map. To easily identify attacks, a "name" attribute is
        computed by concatenating the first letter (or two, if there is a name
        collision) of the following parameters, in order: (1) optimizer, (2)
        random restart used, (3) change of variables applied, (4) loss, (5)
        saliency map, (6) norm used, and (7) jacobian source. Combinations that
        yield known attacks are labeled as such (e.g., gradient descent,
        without random restart or change of variables, categorical
        cross-entropy loss, no saliency map, l∞-norm, with model Jacobians from
        model parameters, will be labeled as "BIM").

        :return: an attack, with a traveler, surface, and saliency map
        :rtype: Attack object
        """

        # instantiate traveler, saliency map, and surface
        self.traveler = traveler.Traveler(
            epochs, optimizer, alpha, random_alpha, change_of_variables
        )
        self.saliency = saliency.SaliencyMap(saliency_map)
        self.surface = surface.Surface(
            model,
            self.saliency,
            loss,
            norm,
            jacobian,
        )

        # save attack parameters
        self.components = {
            "optimizer": optimizer.__name__,
            "random restart": bool(random_alpha),
            "change of variables": change_of_variables,
            "loss function": loss.__name__,
            "saliency map": saliency_map,
            "target norm": f"l{norm}" if type(norm) == int else "l∞",
            "jacobian source": jacobian,
        }

        # build an attack name from the parameters
        name_map = {
            "S-r̶-v̶-Cr-i-∞-m": "BIM",
            "S-R-v̶-Cr-i-∞-m": "PGD",
            "A-r̶-V-CW-i-2-m": "CWL2",
            "S-r̶-v̶-Id-d-2-m": "DF",
            "S-r̶-v̶-Id-j-0-m": "JSMA",
        }
        name = "-".join(
            (
                self.components["optimizer"][0],
                "R" if self.components["random restart"] else "r̶",
                "V" if self.components["change of variables"] else "v̶",
                self.components["loss function"][:2],
                self.components["saliency map"][0],
                self.components["target norm"][1],
                self.components["jacobian source"][0],
            ),
        )
        self.name = name_map.get(name, name)
        return None

    def craft(self, x, y):
        """
        This method crafts adversarial examples, as defined by the instantiated
        Travler, SaliencyMap, and Surface attribute objects.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: n x m tensor
        :param y: the labels (or initial predictions) of x
        :type y: n-length vector
        :return: a batch of adversarial examples
        :rtype: n x m tensor
        """
        x = x.clone()
        self.traveler.initialize(x, self.surface) if self.traveler.init_req else None
        return self.traveler.craft(x, y, self.surface)

    def craft_(self, x, y):
        """
        Similar to the craft method, this method crafts adversarial examples,
        as defined by the instantiated Travler, SaliencyMap, and Surface
        attribute objects. However, akin to PyTorch convention, it does so
        in-place (that is, x is directly modified, instead of a copy of x, as
        is done in the regular craft method).

        :param x: the batch of inputs to produce adversarial examples from
        :type x: n x m tensor
        :param y: the labels (or initial predictions) of x
        :type y: n-length vector
        :return: a batch of adversarial examples
        :rtype: n x m tensor
        """
        self.traveler.initialize(x, self.surface) if self.traveler.init_req else None
        return self.traveler.craft(x, y, self.surface)


def attack_builder(
    epochs=None,
    alpha=None,
    model=None,
    optimizers=(torch.optim.SGD, torch.optim.Adam),
    random_restarts=(False, True),
    apply_change_of_variables=(False, True),
    saliency_maps=("identity", "jsma", "deepfool"),
    norms=(0, 2, float("inf")),
    jacobians=("model",),
    losses=(
        loss.Loss(loss.IdentityLoss, max_obj=False, x_req=False),
        loss.Loss(
            torch.nn.CrossEntropyLoss,
            max_obj=True,
            x_req=False,
            reduction="none",
        ),
        loss.Loss(loss.CWLoss, max_obj=False, x_req=True),
    ),
):
    """
    As shown in [paper_url], seminal attacks in machine learning can be cast
    into a single, unified framework. With this observation, this method
    combinatorically builds attack objects by swapping popular optimizers,
    norms, saliency maps, loss functions, and other "tricks" used within the
    AML community, such as random restart and change of variables. The
    combinations of supported components are shown below:

        optimizers: Adam, Gradient Descent, and Line Search
        random restart: True or False
        change of variables: True or False
        saliency map: Identity, JSMA, or DeepFool
        norm: l₀, l₂, or l∞
        source of model jacobian: model weights or BPDA approach
        loss: Identity, Categorical Cross-Entropy, Hinge or CW

    :param epochs: number of optimization steps to perform
    :type epochs: integer
    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param model: neural network
    :type model: Model object
    :return: a generator yielding attack combinations
    :rtype: generator of Attack objects
    """

    # generate combinations of components and instantiate Attack objects
    num_attacks = (
        len(optimizers)
        * len(random_restarts)
        * len(apply_change_of_variables)
        * len(saliency_maps)
        * len(norms)
        * len(jacobians)
        * len(losses)
    )
    print(f"Yielding {num_attacks} attacks...")
    for (
        optimizer,
        random_restart,
        change_of_variables,
        a_loss,
        saliency_map,
        norm,
        jacobian,
    ) in itertools.product(
        optimizers,
        random_restarts,
        apply_change_of_variables,
        losses,
        saliency_maps,
        norms,
        jacobians,
    ):
        yield Attack(
            epochs=epochs,
            optimizer=optimizer,
            alpha=alpha,
            random_alpha=random_restart * (alpha if alpha else 1),
            change_of_variables=change_of_variables,
            model=model,
            saliency_map=saliency_map,
            loss=a_loss,
            norm=norm,
            jacobian=jacobian,
        )


if __name__ == "__main__":
    """
    Runs all attacks (i.e., all possible combinations in the framework) on
    synthetic data, as shown in [paper_url].
    """
    raise SystemExit(0)
