"""
This module defines custom PyTorch-based loss functions.
Authors: Ryan Sheatsley & Blaine Hoak
Wed Apr 6 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add unit tests to confirm correctness
# optimize cw-loss
# optimize identity loss forward


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    This class is identical to the CrossEntropyLoss class in PyTorch, with the
    exception that two additional attributes are added: max_obj (set as True)
    and req_x (set as False). We add these attributes to let optimizers know
    the direction to step in to produce adversarial examples, as well as to
    notify Attack objects to avoid copying original inputs, as they are not
    needed to compute this loss.

    :func:`__init__` instantiates a CrossEntropyLoss class
    """

    def __init__(self, **kwargs):
        """
        This method instantiates a CrossEntropyLoss object. It accepts keyword
        arguments for the PyTorch parent class described in:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        We note that we explicitly pass reduction="none", as this can cause
        gradients to underflow when computing adversarail examples for large
        batch sizes.

        :param kwargs: keyword arguemtns for torch.nn.CrossEntropyLoss
        :type kwargs: dict
        :return: Cross Entropy loss
        :rtype: CrossEntropyLoss object
        """
        super().__init(reduction="none", **kwargs)
        self.max_obj = True
        self.req_x = False
        return None


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
        This method instantiates a CWLoss object. It accepts three arguments:
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
        self.max_obj = False
        self.req_x = True
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
        prediction. Importantly, this definition can expand the l2-norm vector
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

    def __init__(self):
        """
        This method instantiates an identityloss object. It accepts no
        arguments.

        :return: Identity loss
        :rtype: IdentityLoss object
        """
        super().__init__()
        self.max_obj = False
        self.req_x = False
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


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
