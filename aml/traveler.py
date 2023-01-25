"""
This module defines the travler class referenced in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Wed Apr 27 2022
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

    def __init__(self, change_of_variables, optimizer, random_start):
        """
        This method instantiates Traveler objects with a variety of attributes
        necessary for the remaining methods in this class. Conceptually,
        Travelers are responsible for applying a perturbation to an input based
        on some gradient information, and thus, the following attributes are
        collected: (1) whether change of variables is applied (i.e., mapping
        into and out of the hyperbolic tangent space), (2) a PyTorch-based
        optimizer object from the optimizer module (which is initialized with
        α, the learning rate), (3) the minimum and maximum values to initialize
        inputs (i.e., random start, often uniformly sampled between -ε and ε),
        (4) a tuple of callables to run on the input passed in to __call__, and
        (5) any component that has advertised optimizable hyperparameters.

        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: bool
        :param optimizer: optimization algorithm to use
        :type optimizer: optimizer module object
        :param random_start: desired random start heuristic
        :type random_start: callable
        :param closure: subroutines to run at the end of __call__
        :type closure: tuple of callables
        :return: a traveler
        :rtype: Traveler object
        """
        self.change_of_variables = change_of_variables
        self.optimizer = optimizer
        self.random_start = random_start
        self.closure = [
            comp for c in vars(self) if hasattr(comp := getattr(self, c), "closure")
        ]
        components = (optimizer,)
        self.hparams = dict(*[c.items() for c in components if hasattr(c, "hparams")])
        self.params = {
            "α": optimizer.defaults.get("lr", "adaptive"),
            "CoV": change_of_variables,
            "optim": type(optimizer).__name__,
            "RS": random_start.__name__,
        }
        self.state = False
        return None

    def __call__(self):
        """
        This method is the heart of Traveler objects. It performs two
        functions: (1) to apply a single optimization step (via Optimizer
        objects), and (2) calls closure subroutines before returning. Notably,
        this method assumes that the gradients associated with leaf variables
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

    def initialize(self, x, p, b, norm, eps):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, some attacks (1)
        initialize perturbation vectors via a random perturbation (e.g., PGD &
        FAB), or (2) apply change of variables (e.g. CW) to the original inputs
        which requires that maximum values be less than the minimum value
        mapped to infinity by arctanh (i.e., 1-machine epsilon). Finally, p is
        attached to the optimizer (by reinstantiation, given that PyTorch
        optimizers cannot be initialized without a parameter group).

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :param b: the best perturbation vectors seen thus far
        :type b: torch Tensor object (n, m)
        :param norm: lp norm of the attack
        :type norm: float; 0, 2, or inf
        :param epsilon: maximum allowable distortion
        :type epsilon: float or int if l0
        :return: None
        :rtype: NoneType
        """

        # subroutine (1): random start
        self.random_start(p, b=b, norm=norm, epsilon=eps)

        # subroutine (2): change of variables
        if self.change_of_variables:
            tanh_space(x, True)

        # last subroutine: reinstantiate the optimizer with the perturbation vector
        self.optimizer.__init__([p], **self.optimizer.defaults)
        return None


def identity_start(p, **kwargs):
    """
    This method is an identity function for random start startegies. It returns
    the input as-is. Keyword arguments are accepted to provide a homogeneous
    interface across random start strategies.

    :param p: the perturbation vectors used to craft adversarial examples
    :type p: torch Tensor object (n, m)
    :return: the perturbation vector used to craft adversarial examples
    :rtype: torch Tensor object (n, m)
    """
    return p


def max_start(p, norm, epsilon, **kwargs):
    """
    This function randomly perturbs inputs, made popular by PGD
    (https://arxiv.org/pdf/1706.06083.pdf). The behavior of the random
    perturbation is determined by norm: (1) for l0 attacks, a set of values are
    uniformly sampled from [-1, 1], according to the threat model, (2) for l2
    attacks, values are sampled from a normal distribution and renormed to the
    threat model, and (3) for l∞ attacks, values are sampled uniformly from ±ε.
    Keyword arguments are accepted to provide a homogeneous interface across
    random start strategies.

    :param p: the perturbation vectors used to craft adversarial examples
    :type p: torch Tensor object (n, m)
    :param norm: lp norm of the attack
    :type norm: float; 0, 2, or inf
    :param epsilon: maximum allowable distortion (per input)
    :type epsilon: torch Tensor object (n,)
    :return: randomly perturbed vectors
    :rtype: torch Tensor object (n, m)
    """

    # sample from uniform, select up to l0, and permute indices
    if norm == 0:
        p.uniform_(-1, 1)
        shuffle = torch.randint(p.size(1), p.size()).argsort()
        keep = torch.arange(1, p.size(1) + 1).repeat(p.size(0), 1).add(epsilon)
        return p.scatter_(1, shuffle, p.where(keep > p.size(1), torch.tensor(0)))
    elif norm == 2:
        return p.normal_().renorm_(2, 0, 1).mul_(epsilon)
    elif norm == torch.inf:
        return p.uniform_(-1, 1).clamp_(-epsilon, epsilon)
    else:
        raise NotImplementedError


def shrinking_start(p, b, norm, epsilon, **kwargs):
    """
    This function randomly perturbs inputs based on the norm of the best
    adversarial example seen thus far, made popular by FAB
    (https://arxiv.org/pdf/1907.02044.pdf). Specifically, this function sets
    epsilon to be the minimum of itself and the smallest norm seen thus far,
    divides the result by two, and calls the max_norm function. Keyword
    arguments are accepted to provide a homogeneous interface across random
    start strategies.

    :param p: the perturbation vectors used to craft adversarial examples
    :type p: torch Tensor object (n, m)
    :param b: the best perturbation vectors seen thus far
    :type b: torch Tensor object (n, m)
    :param norm: lp norm of the attack
    :type norm: float; 0, 2, or inf
    :param epsilon: maximum allowable distortion
    :type epsilon: float or int if l0
    :return: randomly perturbed vectors
    :rtype: torch Tensor object (n, m)
    """
    b_norm = b.norm(norm, 1, keepdim=True)
    epsilon = b_norm.where(b_norm < epsilon, torch.tensor(epsilon)).div(2)
    return max_start(p, norm, epsilon.int() if norm == 0 else epsilon)


def tanh_space(x, into=False):
    """
    This function maps x into the tanh-space, as shown in
    https://arxiv.org/pdf/1608.04644.pdf. Specifically, the transformation is
    defined as follows:

                        w = ArcTanh(x * 2 - 1)                          (1)

    where w is x in tanh-space, x is the original input, and Δ is the
    resultant perturbation to be added to x to produce the adversarial
    example (i.e., the variable we optimize over). Upon initialization, we
    first map x into the tanh space as shown in (1), and subsequently, when
    optimizing for Δ, we map x back out of the tanh space via:

                        x = (Tanh(w + Δ) + 1) / 2                       (2)

    where Δ is the computed perturbation to produce an adversarial examples. As
    described in https://arxiv.org/pdf/1608.04644.pdf, (2) ensures that x is
    gauranteed to be within the range [0, 1] (thus avoiding any clipping
    mechanisms). Whether (1) or (2) is applied is determined by the inverse
    argument.

    :param x: the batch of inputs to map into tanh-space
    :type x: torch Tensor object (n, m)
    :param into: whether to map into (or out of) the tanh-space
    :type into: bool
    :return: x mapped into (or back out of) tanh-space
    :rtype: torch Tensor object (n, m)
    """
    eps = 1 - torch.finfo(x.dtype).eps
    return x.mul_(2).sub_(1).mul_(eps).arctanh_() if into else x.tanh().add(1).div(2)


def tanh_space_p(x, p, into=False):
    """
    This function extracts the perturbation vector when using change of
    variables. When using the technique, the computed perturbation is in the
    tanh-space (which is convenient when returning adversarial examples
    directly). To properly compute progress statistics as well as projecting
    onto l2-norm balls (parameterized by epsilon), this method extracts the
    perturbation vector in the real space by first mapping the current input
    out of the tanh-space and subtracting it from the original input. Moreover,
    to facilitate mapping perturbations into the tanh-space, this also supports
    an out-of-place-non-scaled version of the procedure defined in tanh_map.

    :param x: initial inputs in tanh-space
    :type x: torch Tensor object (n, m)
    :param p: perturbations in tanh-space
    :type p: torch Tensor object (n, m)
    :param into: whether to map into (or out of) the tanh-space
    :type into: bool
    :return: perturbation vector mapped out of tanh-space
    :rtype: torch Tensor object (n, m)
    """
    eps = 1 - torch.finfo(x.dtype).eps
    return (
        tanh_space(x).add_(p).mul_(2).sub_(1).mul_(eps).arctanh_().sub_(x)
        if into
        else tanh_space(x.add(p)).sub_(tanh_space(x))
    )
