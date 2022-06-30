"""
This module defines the surface class referenced in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Thu June 30 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# implement BPDA
# track classes and other metadata on initilization of craft?


class Surface:
    """
    This surface class serves to support computing gradients from arbitrary
    combinations of loss functions, saliency maps, and lp-norms. Moreover, as a
    consequence, it exposes models themselves as attributes so that various
    hooks can be implemented to retrieve the current model loss, etc. The
    methods defined within the class are designed to faciliate a gradient-like
    abstraction for PyTorch-based optimizers (as described in the traveler
    module).

    :func:`__init__`: instantiates Surface objects
    :func:`__call__`: performs a single forwards & backwards pass
    :func:`__repr__`: returns Surface parameter values
    :func:`initialize`: prepares Surface objects to operate over inputs
    """

    def __init__(
        self,
        model,
        saliency,
        loss=loss.Loss(
            torch.nn.CrossEntropyLoss,
            max_obj=True,
            x_req=False,
            reduction="None",
        ),
        norm=float("inf"),
        jacobian="model",
        top_n=0.01,
    ):
        """
        This method instantiates a surface object with a variety of attributes
        necessary for supporting (1) arbitrary objective functions, (2) proper
        gradient computation for optimization problems that are defined to
        maximize or minimize objecitve functions, (3) support for arbitrary
        gradient manipulation heuristics (i.e., saliency maps from the Saliency
        module) and (4) casting gradients into the appropriate lp-norm space
        (if the objective function does not explicitly rely on a norm) as to
        abstract such behavior from arbitrary optimizers.

        :param model: neural network
        :type model: Model object
        :param loss: objective function to differentiate
        :type loss: Loss object
        :param saliency: desired saliency map heuristic
        :type saliency: Saliency object
        :param norm: lp-space to project gradients into
        :type norm: int; one of: 0, 2, or float("inf")
        :param jacobian: source to compute jacobian from
        :type jacobian: string; one of "model" or "bpda"
        :param top_n: percent perturbations per iteration for l0 attacks
        :type top_n: float in (0,1]
        :return: a surface
        :rtype: Surface object
        """
        self.model = model
        self.saliency = saliency
        self.loss = loss
        self.norm = norm
        self.nmap = {float("inf"): self.linf, 0: self.ltop, 2: self.l2}
        self.jacobian = jacobian

        # convert top n percentage to discrete based on dimensionality
        self.n = -(-model.features // int(1 / top_n))
        return None

    def backward(self, x, y, cov=False):
        """
        This function computes the gradient of the loss, with respect to x.
        First, we need to check if the application of the saliency map (in step
        two) requires a Jacobian. Since PyTorch does not natively support
        computing Jacobians, we must perform some setup so that it can be
        properly computed efficiently. The setup involes creating n-copies of
        the input x, where n is the number of classes, and performing
        Specifically, it first computes:

                    x.grad = self.loss(self.model(x), y).backward()

        Which populates x.grad with the gradient of the loss. Then, a saliency
        map can be applied to the gradient if so desired:

                        x.grad = self.saliency.apply(x.grad)

        Afterwards, if definition of loss does not already have an lp-norm,
        x.grad is projected into the appropriate norm space, i.e.:

                            x.grad = self.norm(x.grad)

        At this time, when a PyTorch optimizer performs a step, the appropriate
        perturbation is then applied to x transparently.

        :param x: the batch of inputs to differentiate with respect to
        :type x: n x m tensor
        :param y: the batch of labels/initial predictions of x
        :type y: n-length tensor
        :param cov: if change of variables will be applied
        :type cov: bool
        :return: nothing; x.grad is populated correctly
        return NoneType
        """

        # compute model jacobian
        if self.jacobian == "model":

            # avoid full model jacobian if using identity saliency map
            classes = self.model.model[-1].out_features
            x_jac, y_jac, classes = (
                (x, y, 1)
                if self.saliency.stype == "identity"
                else (
                    x.repeat_interleave(classes, dim=0),
                    torch.arange(classes).repeat(x.size(0)),
                    classes,
                )
            )
            x_jac.requires_grad = True
            # map w back to x and track gradients
            # x_j = x_jac.tanh().add(1).div(2) if cov is True else x_jac
            # reattach transformed x to the loss
            # (skip every class-row if saliency is not identity)
            # if self.saliency.stype != "identity":
            #     self.loss.x = x_j[::classes]
            # else:
            #     self.loss.x = x_j
            logits = self.loss(self.model(x_jac), y_jac) * (
                -1 if self.loss.max_obj else 1
            )
            logits.backward(torch.ones(x_jac.size(0)))
            x_jac.requires_grad = False
        else:
            raise NotImplementedError(self.jacobian)

        if cov is True:
            x_jac.grad *= (
                1 - torch.tanh(torch.arctanh(0.999999 * (2 * x_jac - 1))) ** 2
            ) / 2
        # apply saliency map (and norm, if applicable) to the gradient
        saliency_scores = self.saliency.map(
            x_jac.grad.view(-1, classes, x_jac.size(1)),
            logits.view(-1, classes),
            y,
            self.norm,
        )
        x.grad = (
            saliency_scores
            if ((self.saliency.applies_norm) and (self.norm == 2))
            else self.nmap[self.norm](saliency_scores, x)
        )
        return None


def linf(gradient, x):
    """
    This function projects a gradient into the l∞-norm space. The projection is
    simple in that the sign of the gradient is returned. In this way, the
    magnitude of the perturbation is then directly controlled by the learning
    rate of the attached optimizer.

    :param gradient: batch of gradients to project
    :type gradient: n x m tensor
    :param x: current input associated with the gradient
    :type x: n x m tensor
    :return: gradients projected into l-infinity-norm space
    :rtype: n x m tensor
    """
    return torch.sign(gradient)

    def ltop(self, gradient, x):
        """
        This method projects a gradient into a blend of the l0- and l∞-norm
        spaces.  We observe that the conservative nature of l0-targetted
        attacks aids in finding adversarial examples with minimal budget,
        however, the associated computational complexity can be
        cost-prohibitive for datasets of even moderate dimensionality.
        Conversely, l∞-targeted attacks find adversarial examples relatively
        fast, at the cost of exhausting l0-budgets (and having relatively high
        l2-budgets). This projection aims to achieve the best of both worlds,
        by perturbing the top n features, as measured by the gradients, much
        like the Constrained Saliency Projection (CSP) shown in
        https://arxiv.org/pdf/2105.08619.pdf.

        The same mechanisms to address l0-based optimization limitations are
        included in this projection as well.

        :param gradient: batch of gradients to project
        :type gradient: n x m tensor
        :param x: current input associated with the gradient
        :type x: n x m tensor
        :param n: number of features to perturb
        :type n: integer in [0, m]
        :return: gradients projected into l0-norm space
        :rtype: n x m tensor
        """

        # address clipping deficiency
        zeros_map = (x < 1) * (gradient < 0)
        ones_map = (x > 0) * (gradient > 0)
        top_n = torch.topk(torch.abs(gradient) * (zeros_map + ones_map), self.n, dim=1)
        grad_map = (
            torch.scatter(
                torch.zeros_like(gradient),
                dim=1,
                index=top_n.indices,
                src=top_n.values,
            )
            * gradient.sign()
        )

        return torch.sign(grad_map)

    def l0(self, gradient, x):
        """
        This method projects a gradient into the l0-norm space. Ostenisibly,
        this projection serves as a "filter", where the component with the
        largest magnitude is set to the sign of the component times one, while
        all other components are set to zero. The implication being that an
        optimizer who steps in the direction of this gradient will perturb one
        and only one feautre, which is compliant with algorithms that optimize
        over the l0-norm.

        Moreover, l0-based optimization can face a "clipping" deficiency where
        the optimal perturbation aims to go in a direction for a feature whose
        value is already at the maximal or minimal clipping value. Thus,
        post-clip, the feature value remains the same and the optimization
        algorithm fails to move the input further. To address this limitation,
        we ensure that features that are already at the edge of a clip by
        zeroing out the gradients associated with such features (since PyTorch
        optimizers apply perturbations via subtraction, features whose values
        are zero and gradients are positive or values are one and gradients
        negative have their gradients set to zero).

        :param gradient: batch of gradients to project
        :type gradient: n x m tensor
        :param x: current input associated with the gradient
        :type x: n x m tensor
        :return: gradients projected into l0-norm space
        :rtype: n x m tensor
        """

        # address clipping deficiency
        zeros_map = (x < 1) * (gradient < 0)
        ones_map = (x > 0) * (gradient > 0)
        max_map = torch.max(torch.abs(gradient) * (zeros_map + ones_map), dim=1)
        grad_map = (
            torch.scatter(
                torch.zeros_like(gradient),
                dim=1,
                index=max_map.indices.unsqueeze(1),
                src=max_map.values.unsqueeze(1),
            )
            * gradient.sign()
        )

        return torch.sign(grad_map)

    def l2(self, gradient, x, min_l2=torch.tensor(1e-8)):
        """
        This method projects a gradient into the l2-norm space. This projection
        follows the direct definition of computing the l2-norm of a vector.
        Importantly, unlike l0 and l-infinity, the l2-norm encodes magntidue
        directly, and thus, it is sensible for the learning rate of the
        attached optimizer to always be one when perturbing based on the
        l2-norm.

        :param gradient: batch of gradients to project
        :type gradient: n x m tensor
        :param x: current input associated with the gradient
        :type x: n x m tensor
        :param min_l2: minimum l2-norm to scale by (prevents underflow)
        :type min_l2: float
        :return: gradients projected into l0-norm space
        :rtype: n x m tensor
        """
        return gradient / torch.max(
            torch.linalg.norm(gradient, 2, dim=1, keepdim=True), min_l2
        )


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
