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

    def __init__(self, loss, model, norm, saliency_map):
        """
        This method instantiates Surface objects with a variety of attributes
        necessary for the remaining methods in this class. Conceputally,
        Surfaces are responsible for producing gradients from inputs, and thus,
        the following attributes are collected: (1) a PyTorch-based loss object
        from the loss module, (2) a PyTorch-based model object from dlm
        (https://github.com/sheatsley/models), (3) the lp-norm to project
        gradients into, and (4) the saliency map to apply.

        :param loss: objective function to differentiate
        :type loss: loss module object
        :param model: feedforward differentiable neural network
        :type model: dlm LinearClassifier-inherited object
        :param norm: lp-space to project gradients into
        :type norm: surface module object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: surface module object
        :return: a surface
        :rtype: Surface object
        """
        components = (loss, norm, saliency_map)
        self.loss = loss
        self.model = model
        self.norm = norm
        self.saliency_map = saliency_map
        self.closure = [c for c in components if hasattr(c, "closure")]
        self.hparams = dict(c.hparam for c in components if hasattr(c, "hparam"))
        self.params = {
            "loss": type(loss).__name__,
            "model": type(model).__name__,
            "lp": type(norm).__name__,
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

        # apply saliency map and lp-norm gradient projection
        smap_args = {"loss": loss, "y": y, "p": p}
        smap_grad = self.saliency_map(grad.view(-1, cj, grad.size(1)), **smap_args)
        final_grad = self.norm(smap_grad, p=p)

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
        :type clip: tuple of torch Tensor objects (n, m)
        :param p: the perturbation vectors used to craft adversarial examples
        :type p: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """

        # subroutine (1): initialize L0 objects with minimum and maximum values
        self.norm.initialize(clip) if type(self.norm) is L0 else None

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

    def __init__(self, p, classes, minimum=1e-4, **kwargs):
        """
        This method instantiates a DeepFoolSaliency object. As described above,
        ith class is defined as the minimum logit difference scaled by the
        q-norm of the logit differences. Thus, upon initilization, this class
        saves q, the number of classes (so that the yth gradient can be
        retrieved correctly with small batches), and budget and the minimum
        possible norm value (used for mitigating underflow).

        :param p: the lp-norm to apply
        :return: a DeepFool saliency map
        :param classes: number of classes
        :type classes: int
        :param minimum: minimum gradient value (to mitigate underflow)
        :type minimum: float
        :rtype: DeepFoolSaliency object
        """
        self.q = 1 if p == torch.inf else p
        self.classes = classes
        self.minimum = minimum
        return None

    def __call__(self, g, loss, y, p, **kwargs):
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
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: gradient differences as defined by DeepFool
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
        grad_diffs = yth_grad.sub(other_grad)
        logit_diffs = yth_logit.sub(other_logits)
        normed_ith_logit_diff, i = (
            logit_diffs.abs()
            .div(grad_diffs.norm(self.q, dim=2).clamp_(self.minimum))
            .add_(self.minimum)
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
            .div_(ith_grad_diff.pow(self.q).sum(1, keepdim=True).clamp_(self.minimum))
            .add_(self.minimum)
            .mul(ith_grad_diff)
        )
        return ith_grad_diff

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
        return g.mul_(self.normed_ith_logit_diff)


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

        # get yth & ith rows and add stability when using unique losses
        yth = g[torch.arange(g.size(0)), y, :]
        ith = g.sum(1).sub_(yth)
        ith = ith.where(ith.sum(1, keepdim=True) != 0, yth.sign().mul(-1))

        # zero out components whose yth and ith signs are equal and compute product
        smap = (yth.sign() != ith.sign()).float().mul_(yth).mul_(ith.abs_())
        return smap


class L0:
    """
    This class implements methods for projecting gradients into the l0 space
    and enforcing that perturbations are compliant with l0-based threat models.
    For projecting gradients, features are are selected based on a score
    computed from the product of the distance remaining to the bounds of the
    domain and the gradients. For enforcing threat models, once perturbation
    vectors are at the maximum budget, the distance for unperturbed features is
    set to zero (and thus, their scores become zero).

    :func:`__init__`: instantiates L0 objects.
    :func:`__call__`: projects gradients into l0 space
    :func:`initialize`: initializes clipping distance & perturbation amount
    :func:`project`: enforces l0 threat models
    """

    def __init__(self, epsilon, maximize, top=0.01, **kwargs):
        """
        This method instantiates an L0 object. It accepts as arguments the l0
        budget, whether the loss is to be maximized, and the amount of features
        to perturb at each iteration. Keyword arguments are accepted to provide
        a homogeneous interface across norm projection objects.

        :param epsilon: the maximum amount of perturable features
        :type epsilon: int
        :param clip: the initial perturbation range
        :type clip: tuple of torch Tensor objects (n, m)
        :param maximize: whether the loss function is to be maximized
        :type maximize: bool
        :param top: percentage of features to perturb per iteration
        :type top: float
        :return: l0 projection methods
        :rtype: L0 object
        """
        self.epsilon = epsilon
        self.direction = -1 if maximize else 1
        self.top = top
        return None

    def __call__(self, g, p, **kwargs):
        """
        This method projects gradients into the l0-norm space. Specifically,
        features are scored based on the product of their gradients and the
        distance remaining to either the minimum or maximum clipping bounds
        (depending on the sign of the gradient). The top% of components
        (measured by magnitude) are set to their signs, while all other
        component gradients are set to zero.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, m)
        :param p: the current perturbation vector
        :type p: torch Tensor object (n, m)
        :return: gradients projected into l0 space.
        :rtype: torch Tensor object (n, m)
        """
        min_clip = g.sign().mul_(self.direction).eq(1)
        distance = self.clip.sub(p).abs_()
        g.mul_(torch.where(min_clip, *distance.unbind()))
        bottom_k = g.abs().topk(self.bottom, dim=1, largest=False)
        return g.scatter_(dim=1, index=bottom_k.indices, value=0).sign_()

    def initialize(self, clip):
        """
        This method initializes L0 objects by attaching the distance from
        initial inputs to clipping bounds and computes the amount of
        perturbable features per iteration.

        :param clip: the initial allowable perturbation range of the domain
        :type clip: tuple of torch Tensor objects (n, m)
        :rtype: NoneType
        """
        self.clip = torch.stack(clip)
        self.bottom = int(clip[0].size(1) * (1 - self.top))
        return None

    def project(self, p):
        """
        This method projects perturbations so that they are compliant with the
        parameterized l0 threat model. Specifically, this method keeps the
        top-epsilon feature values (sorted by magnitude) and sets the remainder
        to zero. As an additional optimization, this method also sets the
        distance clipping distance to the current value of the perturbation
        vector as to constrain subsequent perturbations so that they are
        complaint with the threat model.

        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: l0-complaint adversarial examples
        :rtype: torch Tensor object (n, m)
        """
        bottom_k = p.abs().topk(p.size(1) - self.epsilon, dim=1, largest=False)
        p.scatter_(dim=1, index=bottom_k.indices, value=0)
        idx = p.eq(0).logical_and_(p.norm(0, dim=1, keepdim=True).eq_(self.epsilon))
        self.clip[idx.expand(2, -1, -1)] = p[idx].repeat(2)
        return p


class L2:
    """
    This class implements methods for projecting gradients into the l2 space
    and enforcing that perturbations are compliant with l2-based threat models.
    For gradient projection, gradients are normalized by their l2 norms. For
    enforcing threat models, perturbation vectors are renormed such that they
    are no greater than the parameterized l2 budget.

    :func:`__init__`: instantiates L2 objects.
    :func:`__call__`: projects gradients into l2 space
    :func:`project`: enforces l2 threat models
    """

    def __init__(self, epsilon, minimum=1e-12, **kwargs):
        """
        This method instantiates an L2 object. It accepts as arguments the l2
        budget and the minimum possible norm value (used for mitigating
        underflow). Keyword arguments are accepted to provide a homogeneous
        interface across norm projection classes.

        :param epsilon: the maximum l2 distance for perturbations
        :type epsilon: float
        :param minimum: minimum norm value (to mitigate underflow)
        :type minimum: float
        :return: l2 projection methods
        :rtype: L2 object
        """
        self.epsilon = epsilon
        self.minimum = minimum
        return None

    def __call__(self, g, **kwargs):
        """
        This method  projects gradients into the l2-norm space. Specifically,
        this is defined as normalizing the gradients by thier l2-norm. Keyword
        arguments are accepted to provide a homogeneous interface across norm
        projection classes.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, m)
        :return: gradients projected into l2 space.
        :rtype: torch Tensor object (n, m)
        """
        return g.div_(g.norm(2, dim=1, keepdim=True).clamp_(self.minimum))

    def project(self, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l2 threat model (i.e., epsilon). Specifically,
        perturbation vectors whose l2-norms exceed the threat model are
        projected back onto the l2-ball.

        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: l2-complaint adversarial examples
        :rtype: torch Tensor object (n, m)
        """
        return p.renorm_(2, dim=0, maxnorm=self.epsilon)


class Linf:
    """
    This class implements methods for projecting gradients into the l∞ space
    and enforcing that perturbations are compliant with l∞-based threat models.
    For gradient projection, gradients are set to their signs. For enforcing
    threat models, perturbation vectors are clamped to ±ε.

    :func:`__init__`: instantiates L∞ objects.
    :func:`__call__`: projects gradients into l∞ space
    :func:`project`: enforces l∞ threat models
    """

    def __init__(self, epsilon, **kwargs):
        """
        This method instantiates an Linf object. It accepts as arguments the l∞
        budget. Keyword arguments are accepted to provide a homogeneous
        interface across norm projection classes.

        :param epsilon: the maximum l∞ distance for perturbations
        :type epsilon: float
        :return: l∞ projection methods
        :rtype: Linf object
        """
        self.epsilon = epsilon
        return None

    def __call__(self, g, **kwargs):
        """
        This function projects gradients into the l∞-norm space. Specifically,
        this is defined as taking the sign of the gradients. Keyword arguments
        are accepted to provide a homogeneous interface across norm projection
        classes.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, m)
        :return: gradients projected into the l∞-norm space
        :rtype: torch Tensor object (n, m)
        """
        return g.sign_()

    def project(self, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l∞ threat model (i.e., epsilon). Specifically,
        perturbation vectors whose l∞-norms exceed the threat model are
        projected back onto the l∞-ball. This is done by clipping perturbation
        vectors by ±ε.

        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: l∞-complaint adversarial examples
        :rtype: torch Tensor object (n, m)
        """
        return p.clamp_(-self.epsilon, self.epsilon)
