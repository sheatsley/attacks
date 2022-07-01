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


def linf(p):
    """
    This function projects gradients into the l∞-norm space. Specifically, this
    is defined as taking the sign of the gradients.

    :param p: the perturbation vector with gradients
    :type p: PyTorch FloatTensor object (n, m)
    :return: gradients projected into the l∞-norm space
    :rtype: PyTorch FloatTensor object (n, m)
    """
    return p.grad.sign_()


def l0(p, clip, max_obj=True):
    """
    This function projects gradients into the l0-norm space. Specifically, this
    is defined as taking the sign of the gradient component with the largest
    magnitude (or top 1% of components, if there are more than 100 features)
    and setting all other component gradients to zero. In addition, this
    function zeroes any component whose direction would result in a
    perturbation that exceeds the valid feature range defined by the clip
    argument (otherwise optimizers would indefinitely perturb the same
    feature).

    :param p: the perturbation vector with gradients
    :type p: PyTorch FloatTensor object (n, m)
    :param clip: the range of allowable values for the perturbation vector
    :type clip: tuple of PyTorch FloatTensor objects (samples, features)
    :param max_obj: whether the used loss function is to be maximized
    :type max_obj: boolean
    :return: gradients projected into the l0-norm space
    :rtype: PyTorch FloatTensor object (n, m)
    """
    valid_components = torch.logical_and(
        *((p != c) or (p.grad.sign() != c.sign() * 1 if max_obj else -1) for c in clip)
    )
    bottom_k = p.grad.mul(valid_components).topk(
        int(p.size(1) * 0.99), dim=1, largest=False
    )
    return p.grad.scatter_(dim=1, index=bottom_k.indices, src=0.0).sign_()


def l2(p, minimum=torch.tensor(1e-8)):
    """
    This function projects gradients into the l2-norm space. Specifically, this
    is defined as normalizing the gradients by thier l2-norm. The minimum
    optional argument can be used to mitigate underflow.


    :param p: the perturbation vector with gradients
    :type p: PyTorch FloatTensor object (n, m)
    :return: gradients projected into the l2-norm space
    :rtype: PyTorch FloatTensor object (n, m)
    """
    return p.grad.div_(p.grad.norm(2, dim=1, keepdim=True)).clamp_(minimum)


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
