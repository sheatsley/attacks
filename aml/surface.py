"""
This module defines the surface class referenced in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Thu June 30 2022
"""
import aml.traveler as traveler  # PyTorch-based custom optimizers for crafting adversarial examples
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class Surface:
    """
    The Surface class handles input gradient computation and manipulation via
    arbitrary combinations of loss functions, models, saliency maps, and
    lp-norms. Under the hood, Surface objects serve as intelligent wrappers for
    Pytorch LinearClassifiers and the methods defined within this class are designed to
    facilitate crafting adversarial examples.

    :func:`__init__`: instantiates Surface objects
    :func:`__call__`: performs a single forwards & backwards pass
    :func:`__repr__`: returns Surface parameter values
    :func:`initialize`: prepares Surface objects to operate over inputs
    """

    def __init__(self, loss, model, norm, saliency_map, change_of_variables=False):
        """
        This method instantiates Surface objects with a variety of attributes
        necessary for the remaining methods in this class. Conceputally,
        Surfaces are responsible for producing gradients from inputs, and thus,
        the following attributes are collected: (1) a PyTorch-based loss object
        from the loss module, (2) a PyTorch-based model object from dlm
        (https://github.com/sheatsley/models), (3) the lp-norm to project
        gradients into, (4) the saliency map to apply, (5) a tuple of callables
        to run on the input passed in to __call__. Notably, the
        change_of_variables argument (configured in the traveler module)
        determines if inputs should be mapped out of the tanh-space before
        passed into models, and (6) any component that has advertised
        optimizable hyperparameters.


        :param loss: objective function to differentiate
        :type loss: loss module object
        :param model: feedforward differentiable neural network
        :type model: dlm LinearClassifier-inherited object
        :param norm: lp-space to project gradients into
        :type norm: surface module callable
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: saliency module object
        :param change_of_variables: whether to map inputs out of tanh-space
        :type change_of_variables: bool
        :return: a surface
        :rtype: Surface object
        """
        self.loss = loss
        self.model = model
        self.norm = norm
        self.saliency_map = saliency_map
        self.cov = traveler.tanh_space if change_of_variables else lambda x: x
        self.closure = [
            comp for c in vars(self) if hasattr(comp := getattr(self, c), "closure")
        ]
        components = (loss, model, saliency_map)
        self.hparams = dict(
            *[c.hparams.items() for c in components if hasattr(c, "hparams")]
        )
        self.params = {
            "loss": type(loss).__name__,
            "model": type(model).__name__,
            "lp": norm.__name__,
            "smap": type(saliency_map).__name__,
        }
        return None

    def __call__(self, x, y, p):
        """
        This method is the heart of Surface objects. It performs four
        functions: it (1) computes the gradient of the loss, with respect to
        the input (e.g., a perturbation vector) while expanding the input by a
        number-of-classes factor (largley based on
        https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa), (2)
        manipulates the computed gradients as defined by a saliency map, (3)
        projects the resulant gradients into the specified lp-space, and (4)
        calls closure subroutines before returning. Notably, this method
        assumes the input is attached to an optimizer that will use the
        computed gradients.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: PyTorch Tensor object (n,)
        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :return: None
        return NoneType
        """

        # expand inputs if components require a full model jacobian
        xj, yj, pj, cj = (
            (
                x.repeat_interleave(cj, dim=0),
                torch.arange(cj).repeat(x.size(0)),
                p.repeat_interleave(cj, dim=0),
                cj,
            )
            if (cj := self.model.params["classes"] * self.saliency_map.jac_req)
            else (x, y, p, 1)
        )

        # map out of tanh- and [0, 1]-space, perform forward & backward passes
        pj.requires_grad = True
        logits = self.model(self.transform(xj + pj))
        loss = self.loss(logits, yj, y if self.saliency_map.jac_req else None, True)
        (grad,) = torch.autograd.grad(loss, pj, torch.ones_like(loss))
        loss = loss.detach()
        pj.requires_grad = False

        # apply saliency map and lp-norm filter
        opt_args = {"loss": loss.view(-1, cj), "y": y, "p": p}
        smap_grad = self.saliency_map(grad.view(-1, cj, grad.size(1)), **opt_args)
        dist = (c.sub(p).abs() for c in self.clip)
        final_grad = (
            self.norm(smap_grad, dist, self.loss.max_obj)
            if self.norm is l0
            else self.norm(smap_grad)
        )

        # call closure subroutines and attach grads to the perturbation vector
        [comp.closure(final_grad) for comp in self.closure]
        p.grad = final_grad
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of surface components.

        :return: the surface components
        :rtype: str
        """
        return f"Surface({self.params})"

    def initialize(self, ranges, clip, p):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, (1) the minimum and
        maximum values for features are saved so that inputs can be
        de-normalized before they are passed into the model, (2) some attacks
        operate under the l0-norm, which exhibit a difficiency when the most
        salient feature is at a clipping or threat-model bound; this can be
        alleviated by considering these bounds when building the saliency map,
        and (3) some attacks directly incorporate lp-norms as part of the loss
        function (e.g., CW-L2) which requires access to the perturbation
        vector. At this time, this initialization method attaches feature
        ranges & clips to Surface objects and perturbation vectors to loss
        functions (if required).

        :param ranges: the range of allowable feature values
        :type ranges: tuple of torch
        :param clip: the range of allowable values for the perturbation vector
        :type clip: tuple of torch Tensor objects (n, m) Tensor objects (n, m)
        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """

        # subroutine (1): save feature ranges
        self.mins, self.maxs = ranges

        # subroutine (2): save minimum and maximum values for l0 attacks
        self.clip = clip

        # subroutine (3): attach the perturbation vector to loss objects
        self.loss.attach(p) if self.loss.del_req else None
        return None

    def min_max_scale(self, x):
        """
        This method maps inputs back into their original domains. Semantically
        identical to sklearn preprocessing MinMaxScaler, this method maps
        inputs back into their original domains according to the minimums and
        maximums seen pre-normalized (which are arguments to the initialize
        method). Specifically, this method applies the following:

                        o = x * (max(x) - min(x)) + min(x)

        Where o is the initial input and x is the transformed input Notably,
        all operations are done in-place.

        :param x: normalized inputs
        :type x: torch Tensor object (n, m)
        :return: denormalized inputs
        :rtype: torch Tensor object (n, m)
        """
        return x.mul_(self.maxs.sub(self.mins)).add_(self.mins)

    def transform(self, x):
        """
        This method applies a series of transformations before an input is
        passed into the model. At this item, this supports two transformations:
        (1) mapping inputs out of the tanh-space, and (2) inversing min-max
        normalization. Notably, many techniques assume inputs lie within [0, 1]
        (such as change of variables), and thus, they must additionally be
        mapped out of [0, 1] before computing model logits. Finally, we detach
        from x when mapping out of [0, 1] to ensure such operations are not
        tracked by autograd.

        :param x: current batch of transformed inputs
        :type x: torch Tensor object (n, m)
        :return: current batch of inputs, de-transformed
        :rtype: torch Tensor object (n, m)
        """
        x = self.cov(x)
        self.min_max_scale(x.detach())
        return x


def linf(g):
    """
    This function projects gradients into the l∞-norm space. Specifically, this
    is defined as taking the sign of the gradients.

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :return: gradients projected into the l∞-norm space
    :rtype: torch Tensor object (n, m)
    """
    return g.sign_()


def l0(g, dist, max_obj):
    """
    This function projects gradients into the l0-norm space. Specifically,
    features are scored by the product of their gradients and the distance
    remaining to either the minimum or maximum clipping bounds (depending on
    the sign of the gradient). Afterwards, the component with the largest
    magntiude is set to its sign (or top 1% of components, if there are more
    than 100 features) while all other component gradients are set to zero.

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :param dist: distance remaining for the perturbation vector
    :type dist: tuple of torch Tensor objects (n, m)
    :param max_obj: whether the used loss function is to be maximized
    :type max_obj: bool
    :return: gradients projected into the l0-norm space
    :rtype: torch Tensor object (n, m)
    """
    min_clip = g.sign().mul_(-1 if max_obj else 1).add_(1).div_(2).bool()
    g.mul_(torch.where(min_clip, *dist))
    bottom_k = g.abs().topk(int(g.size(1) * 0.99), dim=1, largest=False)
    return g.scatter_(dim=1, index=bottom_k.indices, value=0).sign_()


def l2(g, minimum=torch.tensor(1e-8)):
    """
    This function projects gradients into the l2-norm space. Specifically, this
    is defined as normalizing the gradients by thier l2-norm. The minimum
    optional argument can be used to mitigate underflow.

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :param minimum: minimum gradient value (to mitigate underflow)
    :type minimum: torch Tensor object (1,)
    :return: gradients projected into the l2-norm space
    :rtype: torch Tensor object (n, m)
    """
    return g.div_(g.norm(2, dim=1, keepdim=True).clamp_(minimum))
