"""
This module defines custom PyTorch-based saliency maps.
Authors: Blaine Hoak & Ryan Sheatsley
Tue Jul 25 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO:
# add unit tests


class DeepFoolSaliency:
    """
    This class casts a portion of the DeepFool attack
    (https://arxiv.org/pdf/1511.04599.pdf) as a saliency map. Specifically,
    DeepFool adds the following perturbation to inputs:

            |f(x + Δ)_i - f(x + Δ)_y| / ||∇f(x + Δ)_i - ∇f(x + Δ)_y||_q^q
                            * |∇f(x + Δ)_i - ∇f(x + Δ)_y|^(q-1)
                                * sign(∇f(x + Δ)_i - ∇f(x + Δ)_y)

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
    gradients by the scaled logit differences. Finally, this class defines the
    jac_req attribute to signal Surface objects that this class expects a full
    model Jacobian.

    :func:`__init__`: instantiates JacobianSaliency objects.
    :func:`__call__`: computes differences and returns gradient differences
    :func:`closure`: applies scaled logit differences
    """

    jac_req = True

    def __init__(self, q):
        """
        This method instantiates a DeepFoolSaliency object. As described above,
        ith class is defined as the minimum logit difference scaled by the
        q-norm of the logit differences. Thus, upon initilization, this class
        saves q as an attribute.

        :param q: the lp-norm to apply
        :return: a Jacobian saliency map
        :rtype: JacobianSaliency object
        """
        self.q = q
        return None

    def __call__(self, g, loss, y, minimum=torch.tensor(1e-8), **kwargs):
        """
        This method applies the heuristic defined above. Specifically, this
        computes the logit and gradient differences between the true class
        and closest non-true class. It saves these differences as attributes
        to be used later during Surface closure subroutines. Finally, it
        returns the gradient-differences to be normalized by the appropriate
        lp-norm function in the surface module.

        :param g: the gradients of the perturbation vector
        :type g: torch Tensor object (n, c, m)
        :param logits: the current loss (or logits) used to compute g
        :type logits: PyTortch FloatTensor object (n,)
        :param y: the labels (or initial predictions) of x
        :type y: PyTorch Tensor object (n,)
        :param minimum: minimum gradient value (to mitigate underflow)
        :type minimum: torch Tensor object (1,)
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: gradient differences as defined by DeepFool
        :rtype: torch Tensor object (n, m)
        """

        # retrieve yth gradient and logit
        y_hot = torch.nn.functional.one_hot(y).bool()
        yth_grad = g[y_hot]
        yth_logit = loss.masked_select(y_hot)

        # retrieve all non-yth gradients and logits
        other_grad = g[~y_hot].view(g.size(0), -1, g.size(2))
        other_logits = loss.masked_select(~y_hot)

        # compute ith class
        grad_diffs = other_grad.sub(yth_grad)
        logit_diffs = other_logits.sub(yth_logit).abs_()
        normed_ith_logit_diff, i = logit_diffs.div(
            grad_diffs.norm(self.q, dim=1).clamp_(minimum)
        ).topk(1, dim=1, largest=False)

        # save normed ith logit differences and return ith gradient differences
        self.normed_ith_logit_diff = normed_ith_logit_diff
        ith_grad_diff = grad_diffs[torch.arange(grad_diffs.size(0)), i.flatten(), :]
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
        return g.mul_(self.ith_logit_diff)


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

    def __init__(self):
        """
        This method instantiates an IdentitySaliency object. It accepts no arguments.

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

         |J_y| * Σ_[i ≠ y] J_i if sign(J_y) ≠ sign(Σ_[i ≠ y] J_i) else 0

    where J is the model Jacobian, y is the true class, and i are all other
    classes. Algorithmically, the Jacobian saliency map aggregates gradients in
    each row (i.e., class) such that each column (i.e., feature) is set equal
    to the negative of the product of the yth row and the sum of non-yth rows
    (i.e., i) if and only if the signs of the yth row and sum of non-yth rows
    is different. Conceptually, this prioritizes features whose gradients both:
    (1) point away from the true class, and (2) point towards non-true classes.
    Finally, this class defines the jac_req attribute to signal Surface objects
    that this class expects a full model Jacobian.

    :func:`__init__`: instantiates JacobianSaliency objects.
    :func:`__call__`: applies a JSMA-like heuristic
    """

    jac_req = True

    def __init__(self):
        """
        This method instantiates a JacobianSaliency object. It accepts no arguments.

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
        :type y: PyTorch Tensor object (n,)
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: JSMA-like manipulated gradients
        :rtype: torch Tensor object (n, m)
        """

        # get yth row and "gather" ith rows by subtracting yth row
        yth_row = g[torch.arange(g.size(0)), y, :]
        ith_row = g.sum(1).sub(yth_row)

        # zero out components whose yth and ith signs are equal and compute product
        smap = (yth_row.sign() != ith_row.sign()).mul(yth_row.abs()).mul(ith_row)
        return smap


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
