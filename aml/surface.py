"""
This module defines the surface class referenced in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Thu June 30 2022
"""
import aml.traveler as traveler  # PyTorch-based custom optimizers for crafting adversarial examples
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO:
# add unit tests


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
        from the loss module, (2) a PyTorch-based model object from
        scikit-torch (https://github.com/sheatsley/scikit-torch), (3) the
        lp-norm to project gradients into, (4) the saliency map to apply, (5) a
        tuple of callables to run on the input passed in to __call__. Notably,
        the change_of_variables argument (configured in the traveler module)
        determines if inputs should be mapped out of the tanh-space before
        passed into models, and (6) any component that has advertised
        optimizable hyperparameters.


        :param loss: objective function to differentiate
        :type loss: loss module object
        :param model: feedforward differentiable neural network
        :type model: scikit-torch LinearClassifier-inherited object
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
        self.change_of_variables = (
            traveler.tanh_space if change_of_variables else lambda x: x
        )
        self.closure = [
            comp for c in vars(self) if hasattr(comp := getattr(self, c), "closure")
        ]
        components = (loss, model, saliency_map)
        self.hparams = dict(*[c.items() for c in components if hasattr(c, "hparams")])
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
        x_j, y_j, p_j, c_j = (
            (
                x.repeat_interleave(c_j),
                torch.arange(c_j).repeat(x.size(0)),
                p.repeat_interleave(c_j),
                c_j,
            )
            if (c_j := self.model.params["classes"] * self.saliency_map.jac_req)
            else (x, y, p, 1)
        )

        # map out of tanh-space, perform forward & backward passes
        p_j.requires_grad = True
        loss = self.loss(self.model(self.change_of_variables(x_j + p_j)), y_j)
        grads = torch.autograd.grad(loss, p_j, torch.ones_like(loss))[0]
        p_j.requires_grad = False

        # apply saliency map and lp-norm filter
        smap_grads = self.saliency_map(
            grads.view(-1, c_j, grads.size(1)),
            loss=loss.view(-1, c_j),
            y=y,
        )
        final_grads = (
            self.norm(smap_grads, self.clip, self.loss.max_obj)
            if self.norm is l0
            else self.norm(smap_grads)
        )

        # call closure subroutines and attach grads to the perturbation vector
        [comp.closure(final_grads) for comp in self.closure]
        p.grad = final_grads
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of surface components.

        :return: the surface components
        :rtype: str
        """
        return f"Surface({self.params})"

    def initialize(self, clip):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, some attacks operate
        under the l0-norm, which exhibit a difficiency when the most salient
        feature is at a clipping or threat-model bound; this can be alleviated
        by considering these bounds when building the saliency map. At this time,
        this intilization method attaches these bounds to Surface objects.

        :param clip: the range of allowable values for the perturbation vector
        :type clip: tuple of torch Tensor objects (n, m)
        :return: None
        :rtype: NoneType
        """
        self.clip = clip
        return None


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


def l0(g, clip, max_obj):
    """
    This function projects gradients into the l0-norm space. Specifically, this
    is defined as taking the sign of the gradient component with the largest
    magnitude (or top 1% of components, if there are more than 100 features)
    and setting all other component gradients to zero. In addition, this
    function zeroes any component whose direction would result in a
    perturbation that exceeds the valid feature range defined by the clip
    argument (otherwise optimizers would indefinitely perturb the same
    feature).

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :param clip: the range of allowable values for the perturbation vector
    :type clip: tuple of torch Tensor objects (n, m)
    :param max_obj: whether the used loss function is to be maximized
    :type max_obj: bool
    :return: gradients projected into the l0-norm space
    :rtype: torch Tensor object (n, m)
    """
    valid_components = g.mul_(clip[(1 + (g.sign_() * 1 if max_obj else -1)) / 2])
    bottom_k = valid_components.abs_().topk(int(g.size(1) * 0.99), dim=1, largest=False)
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


if __name__ == "__main__":
    """ """
    raise SystemExit(0)