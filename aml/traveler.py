"""
This module defines the traveler class referenced in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Thu Feb 2 2023
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class Traveler:
    """
    The Traveler class handles consuming gradient information from inputs and
    applying perturbations appropriately. Under the hood, Traveler objects
    serve as intelligent wrappers for PyTorch optimizers and the methods
    defined within this class are designed to facilitate crafting adversarial
    examples.

    :func:`__init__`: instantiates Traveler objects
    :func:`__call__`: performs one step of input manipulation
    :func:`__repr__`: returns Traveler parameter values
    :func:`initialize`: prepares Traveler objects to operate over inputs
    """

    def __init__(self, optimizer, random_start):
        """
        This method instantiates Traveler objects with a variety of attributes
        necessary for the remaining methods in this class. Conceptually,
        Travelers are responsible for applying a perturbation to an input based
        on some gradient information, and thus, the following attributes are
        collected: (1) a PyTorch-based optimizer object from the optimizer
        module (which is initialized with α, the learning rate), and (2) the
        random start strategy (if any).

        :param optimizer: optimization algorithm to use
        :type optimizer: optimizer module object
        :param random_start: desired random start heuristic
        :type random_start: callable
        :param closure: subroutines to run at the end of __call__
        :type closure: tuple of callables
        :return: a traveler
        :rtype: Traveler object
        """
        components = (optimizer,)
        self.optimizer = optimizer
        self.random_start = random_start
        self.closure = [c for c in components if hasattr(c, "closure")]
        self.hparams = dict(c.hparam for c in components if hasattr(c, "hparam"))
        self.params = {
            "α": optimizer.defaults.get("lr", "adaptive"),
            "optim": type(optimizer).__name__,
            "rstart": type(random_start).__name__,
        }
        return None

    def __call__(self):
        """
        This method is the heart of Traveler objects. It performs two
        functions: (1) to apply a single optimization step (via Optimizer
        objects), and (2) calls closure subroutines before returning. Notably,
        this method assumes that the gradients associated with parameters
        attached to optimizers is populated.

        :return: None
        :rtype: NoneType
        """
        self.optimizer.step()
        [comp.closure() for comp in self.closure]
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of traveler
        components.

        :return: the traveler components
        :rtype: str
        """
        return f"Traveler({self.params})"

    def initialize(self, p, o):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, (1) some attacks
        initialize perturbation vectors via a random perturbation (e.g., PGD &
        FAB), and (2) p is attached to the optimizer (by reinstantiation, since
        PyTorch optimizers cannot be initialized without a parameter group).

        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :param o: the best perturbation vectors seen thus far
        :type o: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """

        # subroutine (1): random start
        self.random_start(p, o=o)

        # subroutine (2): reinstantiate the optimizer with the perturbation vector
        self.optimizer.__init__([p], **self.optimizer.defaults)
        return None


class IdentityStart:
    """
    This class implements an identity random start strategy. Conceptually, it
    servs as a placeholder for attacks that do not use random start.

    :func:`__init__`: instantiates IdentityStart objects.
    :func:`__call__`: returns the input as-is
    """

    def __init__(self, **kwargs):
        """
        This method instantiates an IdentityStart object. It accepts no
        arguments (keyword arguments are accepted for a homogeneous interface).

        :param norm: lp-norm threat model
        :type norm: 0, 2, or inf
        :param epsilon: maximum allowable distortion
        :type epsilon: float
        :return: an identity random start strategy
        :rtype: IdentityStart object
        """
        return None

    def __call__(self, p, **kwargs):
        """
        This method serves as an identity function. It returns the input as-is.
        Keyword arguments are accepted to provide a homogeneous interface
        across random start strategies.


        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :return: the perturbation vector used to craft adversarial examples
        :rtype: torch Tensor object (n, m)
        """
        return p


class MaxStart:
    """
    This class implements a max start random start strategy, made popular by
    PGD (https://arxiv.org/pdf/1706.06083.pdf). The behavior of the random
    perturbation is determined by norm: (1) for l0 attacks, a set of values are
    uniformly sampled from [-1, 1], according to the threat model, (2) for l2
    attacks, values are sampled from a normal distribution and renormed to the
    threat model, and (3) for l∞ attacks, values are sampled uniformly from ±ε.

    :func:`__init__`: instantiates IdentityStart objects.
    :func:`__call__`: applies a random-start, based on the lp-norm.
    :func:`l0`: applies a random-start for l0-based threat models.
    :func:`l2`: applies a random-start for l2-based threat models.
    :func:`linf`: applies a random-start for l∞-based threat models.
    """

    def __init__(self, norm, epsilon, minimum=1e-12):
        """
        This method instantiates a MaxStart object. It accepts lp-norm threat
        model as an argument to determine which max start strategy to apply.

        :param norm: lp-norm threat model
        :type norm: 0, 2, or inf
        :param epsilon: maximum allowable distortion
        :type epsilon: float
        :param minimum: minimum gradient value (to mitigate underflow in l2)
        :type minimum: float
        :return: a max random start strategy
        :rtype: MaxStart object
        """
        self.norm = norm
        self.epsilon = torch.tensor(epsilon)
        self.minimum = minimum
        self.lp = {0: self.l0, 2: self.l2, torch.inf: self.linf}[norm]
        return None

    def __call__(self, p, **kwargs):
        """
        This method applies the max start random start strategy based on the
        lp-norm passed on initialization. Keyword arguments are accepted to
        provide a homogeneous interface across random start strategies.

        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :return: the perturbation vector used to craft adversarial examples
        :rtype: torch Tensor object (n, m)
        """
        return self.lp(p, self.epsilon)

    def l0(self, p, epsilon):
        """
        This function randomly perturbs inputs, based on the l0-norm budget.
        Semantically, an l0-amount of features are randomly selected whose
        values are then uniformally sampled from [-1, 1]. Computationally, this
        involves sampling all features in the perturbation vector uniformally
        in [-1, 1], then a random permutation of feature indicies is generated,
        and finally, the values of the last ε permuted indices are kept, while
        all other feature values are set back to 0.

        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :param epsilon: maximum allowable distortion (per input)
        :type epsilon: torch Tensor object (n, 1)
        :return: randomly perturbed vectors
        :rtype: torch Tensor object (n, m)
        """
        p.uniform_(-1, 1)
        shuffle = torch.randint(p.size(1), p.size()).argsort()
        keep = torch.arange(1, p.size(1) + 1).repeat(p.size(0), 1).add_(epsilon)
        return p.scatter_(1, shuffle, p.where(keep > p.size(1), torch.tensor(0)))

    def l2(self, p, epsilon):
        """
        This function randomly perturbs inputs, based on the l2-norm budget.
        Specifically, feature values are sampled from a normal distribution and
        normalized such that the l2-norm of the perturbation is equal to ε.

        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :param epsilon: maximum allowable distortion (per input)
        :type epsilon: torch Tensor object (n,)
        :return: randomly perturbed vectors
        :rtype: torch Tensor object (n, m)
        """
        return (
            p.normal_()
            .div_(p.norm(2, dim=1, keepdim=True).clamp_(self.minimum))
            .mul_(epsilon)
        )

    def linf(self, p, epsilon):
        """
        This function randomly perturbs inputs, based on the l∞-norm budget.
        Specifically, feature values are sampled uniformly from ±ε.

        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :param epsilon: maximum allowable distortion (per input)
        :type epsilon: torch Tensor object (n,)
        :return: randomly perturbed vectors
        :rtype: torch Tensor object (n, m)
        """
        return p.uniform_(-1, 1).mul_(epsilon)


class ShrinkingStart(MaxStart):
    """
    This class implements a shrinking-start strategy, made popular by FAB
    (https://arxiv.org/pdf/1907.02044.pdf). Specifically, shrinking start
    randomly perturbs inputs based on the norm fo the best adversarial example
    seen thus far. This class inherits from the MaxStart class as this class
    serves to set epsilon to be: (1) zero, if we have yet to craft any
    adversarial examples, or (2) the minimum of itself and the smallest norm
    seen thus far, divided by two, and subsequently call the appropriate
    MaxNorm subroutine.

    :func:`__init__`: instantiates IdentityStart objects.
    :func:`__call__`: shrinks epsilon and applies a random-start
    """

    def __init__(self, norm, epsilon):
        """
        This method instantiates a ShrinkingStart object. It accepts the
        lp-norm threat model to determine which max start strategy to apply as
        well as the maximum allowable distortion.

        :param norm: lp-norm threat model
        :type norm: 0, 2, or inf
        :param epsilon: maximum allowable distortion
        :type epsilon: float
        :return: a shrinking random start strategy
        :rtype: ShrinkingStart object
        """
        super().__init__(norm, epsilon)
        return None

    def __call__(self, p, o, **kwargs):
        """
        This method shrinks epsilon based on the smallest norm seen thus far
        and the lp-threat model, divided by two. Subsequently, this calls the
        underlying max start lp-specific method with the new epsilon. Keyword
        arguments are accepted to provide a homogeneous interface across random
        start strategies.

        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :param b: the best perturbation vectors seen thus far
        :type b: torch Tensor object (n, m)
        :return: the perturbation vector used to craft adversarial examples
        :rtype: torch Tensor object (n, m)
        """

        # set the "smallest" norm to zero if we have yet to craft
        o = o if o.nan_to_num(posinf=0).nonzero().any() else o.nan_to_num(posinf=0)
        on = o.norm(self.norm, dim=1, keepdim=True)
        epsilon = on.where(on < self.epsilon, self.epsilon).div(2)
        return self.lp(p, epsilon.int() if self.norm == 0 else epsilon)
