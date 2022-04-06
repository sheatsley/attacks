"""
This module defines the saliency class referenced in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Wed Jul 21 2021
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO:
# add comments for unit test


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

    def identity(self, gradient, *args):
        """
        The identity method serves as a NOP for attacks that lack a saliency
        map construction (i.e., attacks that rely on gradient-based information
        of cost functions to apply perturbations). As such, there are two forms
        of implementations to support an identity saliency map: (1) if a model
        jacobian is used as input, then return the the y-truth component, or
        (2) if a cost gradient is used as input, then return it as-is. In this
        implementation, we opt for (2), given that PyTorch is designed for
        computing vector-Jacobian (or Jacobian-vector) products, and thus,
        Jacobian computations are computationally expensive. As shown in
        [paper_url], we measure performance characteristics of attacks, and
        thus, unlike all other methods in this module, we expect a gradient as
        input to this method (Note that the Surface module will still return a
        "Jacobian-like" shape of n x 1 x m, which is why we apply a squeeze
        operation).

        :parm gradient: gradient of a cost function
        :type gradient: n x 1 x m tensor
        :param args: arguments to other methods (to support pipelining)
        :type args: list
        :return: the gradient as-is
        :rtype: n x m matrix
        """
        return gradient.squeeze()

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
    """"""
    jac = torch.randn((15, 3, 5))
    logits = torch.randn((15, 3))
    ytrue = torch.argmax(logits, dim=1)

    sm = SaliencyMap("jsma")
    jsma = sm.map(jac, logits, ytrue, 0)
    sm = SaliencyMap("identity")
    ism = sm.map(jac, logits, ytrue, 1)

    sm = SaliencyMap("deepfool")
    dsm0 = sm.map(jac, logits, ytrue, 0)
    dsm2 = sm.map(jac, logits, ytrue, 2)
    dsminf = sm.map(jac, logits, ytrue, float("inf"))
    raise SystemExit(0)
