"""
This module defines the Surface and Saliency classes referenced in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Thu June 30 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class Surface:
    """
    The Surface class handles input gradient computation and manipulation via
    arbitrary combinations of loss functions, models, saliency maps, and
    lp-norms. Under the hood, Surface objects serve as intelligent wrappers for
    Pytorch LinearClassifiers and the methods defined within this class are
    designed to facilitate crafting adversarial examples.

    :func:`__init__`: instantiates Surface objects
    :func:`__call__`: performs a single forwards & backwards pass
    :func:`__repr__`: returns Surface parameter values
    :func:`initialize`: prepares Surface objects to operate over inputs
    """

    def __init__(self, epsilon, loss, model, norm, saliency_map):
        """
        This method instantiates Surface objects with a variety of attributes
        necessary for the remaining methods in this class. Conceputally,
        Surfaces are responsible for producing gradients from inputs, and thus,
        the following attributes are collected: (1) the lp-based threat model,
        (2) a PyTorch-based loss object from the loss module, (3) a
        PyTorch-based model object from dlm
        (https://github.com/sheatsley/models), (5) the lp-norm to project
        gradients into, and (6) the saliency map to apply.

        :param epsilon: lp threat model (used for l0 attacks)
        :type epsilon: float
        :param loss: objective function to differentiate
        :type loss: loss module object
        :param model: feedforward differentiable neural network
        :type model: dlm LinearClassifier-inherited object
        :param norm: lp-space to project gradients into
        :type norm: surface module callable
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: saliency module object
        :return: a surface
        :rtype: Surface object
        """
        components = (loss, norm, saliency_map)
        self.epsilon = epsilon
        self.loss = loss
        self.model = model
        self.norm = norm
        self.saliency_map = saliency_map
        self.closure = [c for c in components if hasattr(c, "closure")]
        self.hparams = dict(c.hparam for c in components if hasattr(c, "hparam"))
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
        :type y: torch Tensor object (n,)
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

        # perform forward & backward passes
        pj.requires_grad = True
        loss = self.loss(self.model(xj + pj), yj, None if cj == 1 else y)
        (grad,) = torch.autograd.grad(loss, pj, torch.ones_like(loss))
        loss = loss.detach().view(-1, cj)
        pj.requires_grad = False

        # apply saliency map and lp-norm filter
        smap_args = {"loss": loss, "y": y, "p": p}
        smap_grad = self.saliency_map(grad.view(-1, cj, grad.size(1)), **smap_args)
        dist = (c.sub(p).abs_() for c in self.clip)
        lp_args = {"dist": dist, "max_obj": self.loss.max_obj, "epsilon": self.epsilon}
        final_grad = self.norm(smap_grad, **lp_args)

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

    def initialize(self, clip, p):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, (1) some attacks
        operate under the l0-norm, which exhibit a difficiency when the most
        salient feature is at a clipping or threat-model bound; this can be
        alleviated by considering these bounds when building the saliency map,
        and (2) some attacks directly incorporate lp-norms as part of the loss
        function (e.g., CW-L2) which requires access to the perturbation
        vector. At this time, this initialization method attaches clips to
        Surface objects and perturbation vectors to loss functions.

        :param clip: the range of allowable values for the perturbation vector
        :type clip: tuple of torch Tensor objects (n, m) Tensor objects (n, m)
        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """

        # subroutine (1): save minimum and maximum values for l0 attacks
        self.clip = clip

        # subroutine (2): attach the perturbation vector to loss objects
        self.loss.attach(p) if self.loss.p_req else None
        return None


class DeepFoolSaliency:
    """
    This class casts a portion of the DeepFool attack
    (https://arxiv.org/pdf/1511.04599.pdf) as a saliency map. Specifically,
    DeepFool adds the following perturbation to inputs:

      |f(x + Δ)_i - f(x + Δ)_y| / ||∇f(x + Δ)_i - ∇f(x + Δ)_y||_q^q
        * |∇f(x + Δ)_i - ∇f(x + Δ)_y|^(q-1) * sign(∇f(x + Δ)_i - ∇f(x + Δ)_y)

    where f returns the model logits, x is the original input, Δ is the current
    perturbation vector to produce adversarial examples, y is the true class, i
    is next closest class (as measured by logit differences, divided by normed
    gradient differences), ∇f is the gradient of the model with respect to Δ, q
    is defined as p / (p - 1), and p is the desired lp-norm. Algorithmically,
    the DeepFool saliency map computes logit- and gradient-differences, where
    the logit difference (divided by the q-norm of the gradient difference)
    serves as the perturbation strength (i.e., α), while the gradient
    difference serves as the perturbation direction. Moreover, since this
    saliency map requires the gradient differences to be normalized, it
    implements a closure subroutine to multiply the resultant normalized
    gradients by the scaled logit differences.

    Additionally, this class computes the projection above with respect to the
    initial input, as shown in the FAB attack
    (https://arxiv.org/pdf/1907.02044.pdf). Specifically, this entails adding
    the sum of the product of the gradient differences and the current
    perturbation to the logit differences.Finally, this class defines the
    jac_req attribute to signal Surface objects that this class expects a full
    model Jacobian.

    :func:`__init__`: instantiates JacobianSaliency objects.
    :func:`__call__`: computes differences and returns gradient differences
    :func:`closure`: applies scaled logit differences
    """

    jac_req = True

    def __init__(self, p, classes, **kwargs):
        """
        This method instantiates a DeepFoolSaliency object. As described above,
        ith class is defined as the minimum logit difference scaled by the
        q-norm of the logit differences. Thus, upon initilization, this class
        saves q as an attribute as well as the number of classes (so that the
        yth gradient can be retrieved correctly with small batches).

        :param p: the lp-norm to apply
        :return: a DeepFool saliency map
        :param classes: number of classes
        :type classes: int
        :rtype: DeepFoolSaliency object
        """
        self.q = 1 if p == torch.inf else p
        self.classes = classes
        return None

    def __call__(self, g, loss, y, p, minimum=1e-4, **kwargs):
        """
        This method applies the heuristic defined above. Specifically, this
        computes the logit and gradient differences between the true class and
        closest non-true class. It saves these differences as attributes to be
        used later during Surface closure subroutines. Finally, it returns the
        gradient-differences to be normalized by the appropriate lp-norm
        function in the surface module. Notably, this method also computes a
        projection with respect to original input, as used in FAB
        (https://arxiv.org/pdf/1907.02044.pdf), which adds the dot product of
        the gradient differences and current perturbation vector to the logit
        differences.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, c, m)
        :param loss: the current loss (or logits) used to compute g
        :type loss: PyTortch FloatTensor object (n, c)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param minimum: minimum gradient value (to mitigate underflow)
        :type minimum: float
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: absolute gradient differences as defined by DeepFool
        :rtype: torch Tensor object (n, m)
        """

        # retrieve yth gradient and logit
        y_hot = torch.nn.functional.one_hot(y, num_classes=self.classes).bool()
        yth_grad = g[y_hot].unsqueeze_(1)
        yth_logit = loss[y_hot].unsqueeze_(1)

        # retrieve all non-yth gradients and logits
        other_grad = g[~y_hot].view(g.size(0), -1, g.size(2))
        other_logits = loss[~y_hot].view(loss.size(0), -1)

        # compute ith class
        grad_diffs = yth_grad.sub_(other_grad)
        logit_diffs = yth_logit.sub_(other_logits)
        normed_ith_logit_diff, i = (
            logit_diffs.abs()
            .div(grad_diffs.norm(self.q, dim=2).clamp_(minimum))
            .add_(minimum)
            .topk(1, dim=1, largest=False)
        )

        # save ith grad signs & normed logit diffs & return absolute ith gradient diffs
        ith_grad_diff = grad_diffs[torch.arange(grad_diffs.size(0)), i.flatten()]
        ith_logit_diff = logit_diffs.gather(1, i)
        self.normed_ith_logit_diff = normed_ith_logit_diff
        self.ith_grad_diff_sign = ith_grad_diff.sign()

        # compute projection wrt original input to support BackwardsSGD
        pith_logit_diff = ith_grad_diff.mul(p).sum(1, keepdim=True).add_(ith_logit_diff)
        self.org_proj = (
            pith_logit_diff.abs_()
            .div_(ith_grad_diff.pow(self.q).sum(1, keepdim=True))
            .add_(minimum)
            .mul(ith_grad_diff)
        )
        return ith_grad_diff.abs_()

    def closure(self, g):
        """
        This method applies the remaining portion of the DeepFool saliency map
        described above. Specifically, when this method is called, gradients
        are assumed to be normalized via the lp functions within the Surface
        module, and thus, the remaining portion of the DeepFool saliency map is
        to multiply by the resultant gradients by the scaled logit differences
        computed within __call__.

        :param g: the (lp-normalized) gradients of the perturbation vector
        :type g: torch Tensor (n, m)
        :return: finalized gradients for optimizers to step into
        :rtype: torch Tensor (n, m)
        """
        return g.mul_(self.normed_ith_logit_diff).mul_(self.ith_grad_diff_sign)


class IdentitySaliency:
    """
    This class implements an identity saliency map. Conceptually, it serves as
    a placeholder for attacks that do not apply saliency map. As all saliency
    maps anticipate input gradients of shape (samples, classes, features), the
    identity saliency map simply squeezes the classes dimension (which is
    expected to be 1, given that this saliency map does not require a full
    model Jacobian, as defined by the jac_req attribute).

    :func:`__init__`: instantiates IdentitySaliency objects.
    :func:`__call__`: squeezes the classes dimension
    """

    jac_req = False

    def __init__(self, **kwargs):
        """
        This method instantiates an IdentitySaliency object. It accepts no
        arguments (keyword arguments are accepted for a homogeneous interface).

        :return: an identity saliency map
        :rtype: IdentitySaliency object
        """
        return None

    def __call__(self, g, **kwargs):
        """
        This method simply squeezes the classes dimension of the input gradient
        g. To provide a single interface across all saliency maps, keyword
        arguments are also defined.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, c, m)
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: squeezed gradients of the perturbation vector
        :rtype: torch Tensor object (n, m)
        """
        return g.squeeze_()


class JacobianSaliency:
    """
    This class implements a similar saliency map used in the Jacobian-based
    Saliency Map Approach (JSMA) attack, as shown in
    https://arxiv.org/pdf/1511.07528.pdf. Specifically, the Jacobian saliency
    map as used in https://arxiv.org/pdf/2209.04521.pdf is defined as:

         J_y * |Σ_[i ≠ y]| J_i if sign(J_y) ≠ sign(Σ_[i ≠ y] J_i) else 0

    where J is the model Jacobian, y is the true class, and i are all other
    classes. Algorithmically, the Jacobian saliency map aggregates gradients in
    each row (i.e., class) such that each column (i.e., feature) is set equal
    to the product of the yth row and the sum of non-yth rows (i.e., i) if and
    only if the signs of the yth row and sum of non-yth rows is different.
    Conceptually, this prioritizes features whose gradients both: (1) point
    away from the true class, and (2) point towards non-true classes. Finally,
    this class defines the jac_req attribute to signal Surface objects that
    this class expects a full model Jacobian. Notably, in this formulation, the
    absolute value is taken on the sum of non-yth rows (as opposed to the yth
    row, as done in https://arxiv.org/pdf/2209.04521.pdf). This change is to
    provide a conceptually similar saliency map to DeepFool (in that
    subtracting the DeepFool saliency map reduces model accuracy) given that
    PyTorch optimizers now directly support maximizing or minimizing objective
    functions since 1.31.

    :func:`__init__`: instantiates JacobianSaliency objects.
    :func:`__call__`: applies a JSMA-like heuristic
    """

    jac_req = True

    def __init__(self, **kwargs):
        """
        This method instantiates a JacobianSaliency object. It accepts no
        arguments (keyword arguments are accepted for a homogeneous interface).

        :return: a Jacobian saliency map
        :rtype: JacobianSaliency object
        """
        return None

    def __call__(self, g, y, **kwargs):
        """
        This method applies the heuristic defined above. Specifically, this:
        (1) computes the the sum of the gradients for non-true classes and
        zeroes out components whose sum has the same sign as the yth row, (2)
        computes the product of the yth row with non-yth rows, and (3) returns
        the negative of the result. Finally, to provide a single interface
        across all saliency maps, keyword arguments are also defined.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, c, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: JSMA-like manipulated gradients
        :rtype: torch Tensor object (n, m)
        """

        # get yth row and "gather" ith rows by subtracting yth row
        yth = g[torch.arange(g.size(0)), y, :]
        ith = g.sum(1).sub_(yth)

        # compute biased projection & add stability when using unique losses
        self.org_proj = torch.zeros((g.size(0), g.size(2)))
        ith = ith.where(ith.sum(1, keepdim=True) != 0, yth.sign().mul(-1))

        # zero out components whose yth and ith signs are equal and compute product
        smap = (yth.sign() != ith.sign()).float().mul_(yth).mul_(ith.abs_())
        return smap


def l0(g, dist, epsilon, max_obj, top=0.01, **kwargs):
    """
    This function projects gradients into the l0-norm space. Specifically,
    features are scored by the product of their gradients and the distance
    remaining to either the minimum or maximum clipping bounds (depending on
    the sign of the gradient). Afterwards, the minimum of the lp threat model
    and the top% of components (measured by magnitude), are set to their signs,
    while all other component gradients are set to zero. Keyword arguments are
    accepted to provide a homogeneous interface across norm projections.

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :param dist: distance remaining for the perturbation vector
    :type dist: tuple of torch Tensor objects (n, m)
    :param epsilon: lp threat model (used for l0 attacks)
    :type epsilon: float
    :param max_obj: whether the used loss function is to be maximized
    :type max_obj: bool
    :param top: percentage of features to perturb
    :type top: float
    :return: gradients projected into the l0-norm space
    :rtype: torch Tensor object (n, m)
    """
    fix = g.size(1) - max(min(int(g.size(1) * top), epsilon), 1)
    min_clip = g.sign().mul_(-1 if max_obj else 1).eq(1)
    g.mul_(torch.where(min_clip, *dist))
    bottom_k = g.abs().topk(fix, dim=1, largest=False)
    return g.scatter_(dim=1, index=bottom_k.indices, value=0).sign_()


def l2(g, minimum=1e-4, **kwargs):
    """
    This function projects gradients into the l2-norm space. Specifically, this
    is defined as normalizing the gradients by thier l2-norm. The minimum
    optional argument can be used to mitigate underflow. Keyword arguments are
    accepted to provide a homogeneous interface across norm projections.

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :param minimum: minimum gradient value (to mitigate underflow)
    :type minimum: torch Tensor object (1,)
    :return: gradients projected into the l2-norm space
    :rtype: torch Tensor object (n, m)
    """
    return g.div_(g.norm(2, dim=1, keepdim=True).clamp_(minimum))


def linf(g, **kwargs):
    """
    This function projects gradients into the l∞-norm space. Specifically, this
    is defined as taking the sign of the gradients. Keyword arguments are
    accepted to provide a homogeneous interface across norm projections.

    :param g: the gradients of the perturbation vector
    :type g: torch Tensor object (n, m)
    :return: gradients projected into the l∞-norm space
    :rtype: torch Tensor object (n, m)
    """
    return g.sign_()
