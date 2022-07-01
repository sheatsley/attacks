"""
This module defines the travler class referenced in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Wed Apr 27 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add unit tests


class Traveler:
    """
    The Traveler class handles consuming gradient information from inputs and
    applying perturbations appropriately, as shown in [paper_url]. Under the
    hood, Traveler objects serve as intelligent wrappers for PyTorch optimizers
    and the methods defined within this class are designed to facilitate
    crafting adversarial examples.

    :func:`__init__`: instantiates Traveler objects
    :func:`__call__`: performs one step of input manipulation
    :func:`__repr__`: returns Traveler parameter values
    :func:`initialize`: prepares Traveler objects to operate over inputs
    :func:`tanh_space`: maps a tensor into (and out of) tanh-space
    """

    def __init__(self, change_of_variables, optimizer, random_restart, closure=()):
        """
        This method instantiates Traveler objects with a variety of attributes
        necessary for the remaining methods in this class. Conceptually,
        Travelers are responsible for applying a perturbation to an input based
        on some gradient information, and thus the following attributes are
        collected: (1) whether change of variables is applied (i.e., mapping
        into and out of the hyperbolic tangent space), (2) an PyTorch-based
        optimizer object from the optimizer module (which is initialized with
        α, the learning rate), (3) the minimum and maximum values to initialize
        inputs (i.e., random restart, often sampled between -ε and ε ), and (5)
        a tuple of callables to run on the input passed in to __call__.

        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: boolean
        :param optimizer: optimization algorithm to use
        :type optimizer: optimizer module object
        :param random_restart: magnitude of a random perturbation
        :type random_restart: scalar
        :param closure: subroutines to run at the end of __call__
        :type closure: tuple of callables
        :return: a traveler
        :rtype: Traveler object
        """
        self.change_of_variables = change_of_variables
        self.optimizer = optimizer
        self.random_restart = random_restart
        self.closure = closure
        self.params = {
            "α": optimizer.lr,
            "CoV": change_of_variables,
            "optim": optimizer.__name__,
            "RR": (-random_restart, random_restart),
        }
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of traveler
        components.

        :return: the traveler components
        :rtype: str
        """
        return f"Traveler({self.params})"

    def __call__(self):
        """
        This method is the heart of Traveler objects. It performs two
        functions: (1) to perform a single optimization step (via Optimizer
        objects), and (2) calls closure subroutines before returning. Notably,
        this method assumes that the gradients associated with leaf variables
        attached to optimizers is populated.

        :return: None
        :rtype: NoneType
        """
        self.optimizer.step()
        [f() for f in self.closure]
        return None

    def initialize(self, x, p):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, some attacks (1)
        initialize perturbation vectors via a random perturbation (e.g., PGD),
        or (2) apply change of variables (e.g. CW) to the original inputs which
        requires that maximum values be less than the minimum value mapped to
        infinity by arctanh (i.e., 1-machine epsilon). Finally, x is attached
        to the optimizer.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: PyTorch FloatTensor object (n, m)
        :param p: the perturbation vectors used to craft adversarial examples
        :type p: PyTorch FloatTensor object (n, m)
        :return: None
        :rtype: NoneType
        """

        # subroutine (1): random restart
        print(f"Applying random restart {self.params['RR']} to {len(p)} vectors...")
        p.add_(
            torch.distributions.uniform.Uniform(
                -self.random_restart, self.random_restart
            ).sample(p.size())
        )

        # subroutine (2): change of variables
        if self.change_of_variables:
            print(f"Applying change of variables to {len(x)} samples...")
            self.tanh_space(x, True)

        # final subroutine: attach p to the optimizer
        print(f"Attaching the perturbation vector to {self.optmizer.__name__}...")
        self.optimizer.add_param_group({"params": p})
        return None

    def tanh_space(self, x, into=False):
        """
        This method maps x into the tanh-space, as shown in
        https://arxiv.org/pdf/1608.04644.pdf. Specifically, the transformation
        is defined as follows:

                            w = ArcTanh(x * 2 - 1)                          (1)

        where w is x in tanh-space, x is the original input, and Δ is the
        resultant perturbation to be added to x to produce the adversarial
        example (i.e., the variable we optimize over). Upon initialization, we
        first map x into the tanh space as shown in (1), and subsequently, when
        optimizing for Δ, we map x back out of the tanh space via:

                            x = (Tanh(w + Δ) + 1) / 2                       (2)

        where Δ is the computed perturbation to produce an adversarail
        examples. As described in https://arxiv.org/pdf/1608.04644.pdf, (2)
        ensures that x is gauranteed to be within the range [0, 1] (thus
        avoiding any clipping mechanisms). Whether (1) or (2) is applied is
        determined by the inverse argument, and, importantly, this method
        operates in-place, as x often needs to be attached to the optimizer.

        :param x: the batch of inputs to map into tanh-space
        :type x: PyTorch FloatTensor object (n, m)
        :param into: whether to map into (or out of) the tanh-space
        :type into: boolean
        :return: x mapped into (or back out of) tanh-space
        :rtype: PyTorch FloatTensor object (n, m)
        """
        return (
            x.mul_(2).sub_(1).arctanh_(1 - torch.finfo(x.dtype).eps)
            if into
            else x.tanh_().add_(1).div_(2)
        )


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
