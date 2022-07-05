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
    (https://arxiv.org/pdf/1511.04599.pdf) as a saliency map. Specifically, DeepFool
    applies
    """

    jac_req = True


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
        :type g: PyTorch FloatTensor object (n, c, m)
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: squeezed gradients of the perturbation vector
        :rtype: PyTorch FloatTensor object (n, m)
        """
        return g.squeeze_()


class JacobianSaliency:
    """
    This class implements a similar saliency map used in the Jacobian-based
    Saliency Map Approach (JSMA) attack, as shown in
    https://arxiv.org/pdf/1511.07528.pdf. Specifically, the Jacobian saliency
    map as used in [paper_url] is defined as:

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
        :type g: PyTorch FloatTensor object (n, c, m)
        :param y: the labels (or initial predictions) of x
        :type y: PyTorch Tensor object (n,)
        :param kwargs: miscellaneous keyword arguments
        :type kwargs: dict
        :return: JSMA-like manipulated gradients
        :rtype: PyTorch FloatTensor object (n, m)
        """

        # get yth row and "gather" ith rows by subtracting yth row
        yth_row = g[torch.arange(g.size(0)), y, :]
        ith_row = g.sum(1) - yth_row

        # zero out components whose yth and ith signs are equal and compute product
        return -(yth_row.sign() != ith_row.sign()) * yth_row * ith_row


class SaliencyMap:
    """
    The SaliencyMap class serves as a generalized framework for the methods
    implemented acrosss all saliency maps. When used on its own, this class
    operates as an "identity" saliency map by returning an unmodified version
    of what was passed into it. This SaliencyMap is largely for the purposes of
    running an attack that does not inherently use a saliency map while keeping
    a generalized framework for an attack. Moreover, it also contains logic to
    support parameterized exponentiation on the scalar part of the saliency
    map. For the methods defined below, we use n to be the number of samples, c
    to be the number of classes, and m to be the number of features. This class
    contains the following methods:

    :func `__init__`: initializes bookkeeping parameters
    :func `identity`: returns the input as-is (expects a gradient)
    :func `jsma`: computes the saliency used in the Adapative JSMA
    :func `deepfool`: casts the DeepFool attack as a saliency map
    """

    def __init__(self, saliency_type):
        """
        This function initializes the general SaliencyMap class.
        It takes the following arguments:

        :param saliency_type: the type of saliency map to use
        :type saliency_type: string; one of "identity", "jsma", or "deepfool"
        """
        self.applies_norm = True if saliency_type == "deepfool" else False
        self.map = getattr(self, saliency_type)
        self.stype = saliency_type
        return None

    def jsma(self, j, logits, true_class, p):
        """
        *Insert description here*

        :param j: jacobian matrix
        :type j: n x c x m tensor
        :param logits: model logits (or loss output)
        :type logits: n x c matrix
        :param true_class: class labels
        :type true_class: n-length vector
        :param p: the lp-norm to be applied to the saliency map
        :type p: int
        :return: applied AJSMA saliency map
        :rtype: n x m matrix
        """

        # get y true row of jacobian, unsqueeze to nx1, repeat class across m
        # features to get nxm, unsqueeze to get nx1xm
        ytrue = true_class.unsqueeze(1).repeat(1, j.shape[-1]).unsqueeze(1)
        ytruejac = torch.gather(j, dim=1, index=ytrue).squeeze(1)

        # sum all the jacobian class rows, subtract the true class row to get
        # the sum of all nontrue classes
        nontruejac = torch.sum(j, dim=1) - ytruejac

        # since we do not want to pertub if the ytruejac and nontrue jac are
        # the same sign, we can zero this by multiplying by the sign
        # differences (if they are the same, this will be zero) this will also
        # ensure our final signage is correct for the elements we do want to
        # perturb
        signdiff = (torch.sign(ytruejac) - torch.sign(nontruejac)) / 2
        salmap = signdiff * ytruejac * nontruejac
        return -salmap

    def deepfool(self, j, logits, true_class, p=2):
        """
        *Insert description here*

        :param j: jacobian matrix
        :type j: n x c x m tensor
        :param logits: model logits (or loss output)
        :type logits: n x c matrix
        :param true_class: class labels
        :type true_class: n-length vector
        :param p: the lp-norm to be applied to the saliency map
        :type p: int
        :param λ: exponentiation parameter (original attack is λ=1.0)
        :type λ: float
        :param λ_op: sets the saliency map compoentns to be exponentiated
        :type λ_op: string; one of: "scalar", "vector", or "all"
        :return: applied DeepFool saliency map
        :rtype: n x m matrix
        """

        # get y true row of jacobian, unsqueeze to nx1, repeat class across m
        # features to get nxm, unsqueeze to get nx1xm
        ytrue = true_class.unsqueeze(1).repeat(1, j.shape[-1]).unsqueeze(1)
        ytruejac = torch.gather(j, dim=1, index=ytrue)

        # get true logit row, unsqueeze to nx1 and gather
        ytruelogit = torch.gather(logits, dim=1, index=true_class.unsqueeze(1))

        # taking the difference between true and other classes jacobian and
        # logits to determine target class
        jdiff = j - ytruejac
        logitdiff = logits - ytruelogit

        # # taking norm based on p across features and adding small value to avoid div by zero
        # normexp = 1 if p == float("inf") or p == 0 else p
        # jdiffnorm = (
        #     torch.pow(torch.pow(torch.abs(jdiff), normexp).sum(dim=2), 1 / normexp)
        #     + 10e-8
        # )

        # # getting target class based on min distance from true class
        # dist = torch.abs(logitdiff) / jdiffnorm
        # distinf = dist.scatter(
        #     dim=1,
        #     index=true_class.unsqueeze(1),
        #     src=torch.ones_like(dist) * float("inf"),
        # )
        # l = torch.argmin(distinf, dim=1)

        # logitdiff_l = torch.gather(torch.abs(logitdiff), dim=1, index=l.unsqueeze(1))
        # jdiff_l = torch.gather(
        #     jdiff, dim=1, index=l.unsqueeze(1).repeat(1, jdiff.shape[-1]).unsqueeze(1)
        # ).squeeze(1)

        # if p == 2:
        #     salmap = (
        #         jdiff_l
        #         * logitdiff_l
        #         / ((jdiff_l ** 2).sum(dim=1, keepdim=True) + torch.tensor(10e-8))
        #     )
        # else:
        #     # if not the l2 norm, then we take the norm in the surface
        #     salmap = jdiff_l
        # salmap = torch.nan_to_num(salmap, nan=0.0, posinf=None, neginf=None)
        # return -salmap
        # taking norm based on p across features and adding small value to avoid div by zero
        jdiffnorm = (
            torch.linalg.norm(jdiff, ord=1 if p == float("inf") else p, dim=2) + 10e-8
        )

        # getting target class based on min distance from true class
        dist = torch.abs(logitdiff) / jdiffnorm
        distinf = dist.scatter(
            dim=1,
            index=true_class.unsqueeze(1),
            src=torch.ones_like(dist) * float("inf"),
        )
        l = torch.argmin(distinf, dim=1)

        logitdiff_l = torch.gather(torch.abs(logitdiff), dim=1, index=l.unsqueeze(1))
        jdiff_l = torch.gather(
            jdiff, dim=1, index=l.unsqueeze(1).repeat(1, jdiff.shape[-1]).unsqueeze(1)
        ).squeeze(1)

        jdiff_l = torch.pow(torch.abs(jdiff_l), 1.0) * torch.sign(jdiff_l)

        if p == 0:
            # we want to apply l0 norm in the surface instead of here to
            # support a pseudo search space
            """
            feat_map = (
                torch.pow(logitdiff_l, λ if λ_op == "scalar" or "whole" else 1.0)
                / jdiff_l
            )
            max_feat = torch.argmax(torch.abs(feat_map), dim=1)
            salmap = torch.zeros_like(jdiff_l)
            salmap = salmap.scatter(
                dim=1,
                index=max_feat.unsqueeze(1),
                src=torch.max(feat_map, dim=1).values.unsqueeze(1),
            )
            """
            feat_map = torch.pow(jdiff_l, 1.0)
            salmap = feat_map
        elif p == float("inf"):
            # for l-inf, none of the lambda ops will matter since they will not
            # affect the sign
            """
            salmap = torch.nan_to_num(
                logitdiff_l
                / (torch.linalg.norm(jdiff_l, ord=1, dim=1).unsqueeze(1) + 1e-8)
                * jdiff_l,
                nan=0.0,
                posinf=None,
                neginf=None,
            )
            """
            salmap = torch.nan_to_num(jdiff_l)

        else:
            salmap = jdiff_l * torch.pow(
                logitdiff_l
                / torch.max(
                    torch.pow(
                        torch.linalg.norm(jdiff_l, ord=2, dim=1, keepdim=True), 2
                    ),
                    torch.tensor(1e-8),
                ),
                1.0,
            )
            salmap = torch.nan_to_num(salmap, nan=0.0, posinf=None, neginf=None)

        return -salmap


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
