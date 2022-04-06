"""
This module defines custom PyTorch-based loss functions.
Authors: Ryan Sheatsley & Blaine Hoak
Wed Apr 6 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class CWLoss(torch.nn.Module):
    """
    This class implements the Carlini-Wagner loss, as shown in
    https://arxiv.org/pdf/1608.04644.pdf. Specifically, the loss is defined as:

            lp(advx, x) + c * max(max(model(x)_i:i≠y) - model(x)_y, -k)

    where x is the original input, advx is the resultant adversarial example, y
    is the predicted class of x and i is the next closest class (as measured
    through the model logits), lp computes the p-norm of the differecne between
    advx and x, while k and c serve as hyperparameters. k controls how deep
    adversarial examples are pushed across the decision boundary, while c
    parameterizes the tradeoff between imperceptibility and misclassification.
    Note that while the original Carlini-Wagner objective function applies a
    change of variables to x (so that it is easier to find adversarial
    examples), that is a separate component from this loss (see the surface
    module for further details).

    :func:`__init__`: instantiates a cwloss object
    :func:`forward`: returns the loss for a given batch of inputs
    """

    def __init__(self, norm=2, c=1, k=0):
        """
        This method instantiates a cwloss object. It accepts three arguments:
        (1) p, the lp-norm to use, (2) c, which emphasizes optimizing
        adversarial goals over the introduced distortion, and (3) k, which
        controls how far adversarial examples are pushed across the decision
        boundary (as measured through the difference between the initial and
        current predictions).
        :param norm: lp-space to project gradients into
        :type norm: int; one of: 0, 2, or float("inf")
        :param c: encourages misclassification over imperceptability
        :type c: float
        :param k: encourages "deep" adversarial examples
        :type k: float
        :return: Carlini-Wagner loss
        :rtype: CWLoss object
        """
        super().__init__()
        self.norm = norm
        self.c = c
        self.k = k
        return None

    def forward(self, logits, y):
        """
        This method computes the loss described above. Specifically, it
        computes the sum of: (1) the lp-norm of the perturbation, and (2) the
        difference between the logits component corresponding to the initial
        prediction and the maximum logits component not equal to the initial
        prediction. Importantly, this definition  can expand the l2-norm vector
        by a factor of c (where is the number of classes),  depending on if the
        attack parameters require a full Jacobian to be computed (i.e., if a
        saliency map is used), by repeating the inputs by a factor of c as
        well.

        :param logits: the model logits, evaluated at advx
        :type logits: n x c tensor (where c is the number of classes)
        :param y: the labels (or initial predictions) of x
        :type y: n-length vector
        :return: loss values
        :rtype: n-length vector
        """
        lp = torch.linalg.norm(
            self.x - self.original_x, ord=self.norm, dim=1
        ).repeat_interleave(1 if self.x.size(0) == logits.size(0) else logits.size(1))

        # compute misclassification term
        misclass = (
            logits.gather(1, y.view(-1, 1))
            .flatten()
            .sub(
                logits.masked_select(
                    torch.ones(logits.shape, dtype=torch.bool).scatter(
                        1, y.view(-1, 1), 0
                    )
                )
                .view(-1, logits.size(1) - 1)
                .max(1)[0]
            )
            + self.k
        )
        return lp + self.c * misclass


class IdentityLoss(torch.nn.Module):
    """
    This class implements a psuedo-identity loss function. As many attacks in
    adversarial machine learning (such as the JSMA or DeepFool) rely on the
    model Jacobian to apply perturbations coupled with the fact that PyTorch
    cannot compute Jacobians directly, this loss function seeks to emulate
    computing the model Jacobian by simply returning the yth-component of the
    model logits. Thus, when x is duplicated c times, where c is the number of
    classes, backpropogating the gradients from this loss function will return
    a model Jacobian.

    :func:`__init__`: instantiates an identityloss object
    :func:`forward`: returns the yth logit component for a batch of imputs
    """

    def __init__(self, *args, **kwargs):
        """
        This method instantiates an identityloss object. It accepts no
        arguments.

        :return: Identity loss
        :rtype: IdentityLoss object
        """
        super().__init__()
        return None

    def forward(self, logits, y):
        """
        This method computes the loss described above. Specifically, it returns
        the yth-component of the logits.

        :param logits: the model logits, evaluated at advx
        :type logits: n x c tensor (where c is the number of classes)
        :param y: the labels (or initial predictions) of x
        :type y: n-length vector
        :return: loss values
        :rtype: n-length vector
        """
        return logits.gather(1, y.view(-1, 1)).flatten()


def Loss(loss, max_obj, x_req, *args, **kwargs):
    """
    This class extends loss definitions, through dynamic inheritence (for
    losses that subclass _Loss in torch.nn.modules.loss) to contain two
    additional attributes: (1) whether the objective function is to be
    maximized (which requires the signs of the gradients to be flipped, as
    PyTorch optimizers are defined to minimize objecitve functions), and (2)
    whether the loss function requires the original inputs (which is typically
    used for losses that compute lp-norms from the current adversarial
    examples, such as Carlini-Wagner loss). Specifically, (2) is realized by
    setting a copy of the original input, as well as a pointer to the input (to
    be later attached to an optimizer), as attributes, so that norms can be
    readily measured throughout the crafting process.

    :param loss: loss object to inherit from
    :type loss: either a subclass of _Loss or torch.nn.Module
    :param applies_norm: whether loss applies a norm
    :type applies_norm: boolean
    :param args: any arguments for loss
    :type args: list
    :param kwargs: any keyword arguments for loss
    :type kwargs: dictionary
    :return: a Loss object
    :rtype: Loss
    """

    class Loss(loss):
        """
        This class serves a generic loss class that adds any custom attributes
        to the parent class of the object argument.

        :func:`__init__` inherits from loss and instantiates a Loss object
        """

        def __init__(self, max_obj, x_req, *args, **kwargs):
            """
            This method inherits from loss and instantiates a Loss object

            :param max_obj: whether the objective is to maximize
            :type max_obj: boolean
            :param x_req: whether the loss requires the original x
            :type x_req: boolean
            :param args: any arguments for loss
            :type args: list
            :param kwargs: any keyword arguments for loss
            :type kwargs: dictionary
            :return: a Loss object
            :rtype: Loss
            """
            super().__init__(*args, **kwargs)
            self.max_obj = max_obj
            self.x_req = x_req
            self.__name__ = self.__class__.__bases__[0].__name__
            return None

        def __repr__(self):
            """
            Since the Loss class simply attaches an attribute to the parent
            class, we also return the class name of the parent.

            :return: parent class name and arguments
            :rtype: string
            """
            return f"Loss({self.__name__})"

        def attach(self, x):
            """
            As described above, loss functions that applie norms typically
            measure distance between an original input and the resultant
            adversarial example. To this end, this method serves to attach the
            orginal input at craf-time to facilitiate norm computation.
            Specifically, the input x is first cloned and then a reference to x
            is saved (which should eventually be attached to an optimizer); in
            this way, we can compute arbitrary lp-norm as the optimizer
            perturbs x, without explicitly passing x into this loss at runtime.

            :param x: the input to clone and attach
            :type x: n x m tensor
            :return: None
            :rtype: NoneType
            """
            self.original_x = x.clone()
            self.x = x
            return None

    return Loss(max_obj, x_req, *args, **kwargs)


if __name__ == "__main__":
    """
    Example usage from [paper_url].
    """
    raise SystemExit(0)
