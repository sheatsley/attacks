"""
This module defines custom PyTorch-based loss functions.
Authors: Ryan Sheatsley & Blaine Hoak
Wed Apr 6 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add unit tests to confirm correctness
# optimize cw-loss


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    This class is identical to the CrossEntropyLoss class in PyTorch, with the
    exception that: (1) the most recently computed loss is stored in curr_loss
    (to faciliate optimizers who require it) and (2) two additional attributes
    are added: del_req (set to False) and max_obj (set as True). We add these
    attributes to let optimizers know the direction to step in to produce
    adversarial examples, as well as to notify Attack objects that a reference
    to the current perturbation vector is not needed by this class.

    :func:`__init__` instantiates a CrossEntropyLoss object
    """

    del_req = False
    max_obj = True

    def __init__(self, **kwargs):
        """
        This method instantiates a CrossEntropyLoss object. It accepts keyword
        arguments for the PyTorch parent class described in:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        We note that we explicitly pass reduction="none", as this can cause
        gradients to underflow when computing adversarial examples for large
        batch sizes.

        :param kwargs: keyword arguments for torch.nn.CrossEntropyLoss
        :type kwargs: dict
        :return: Cross Entropy loss
        :rtype: CrossEntropyLoss object
        """
        super().__init(reduction="none", **kwargs)
        return None

    def forwad(self, logits, y):
        """
        This method serves as a wrapper for the PyTorch CrossEntropyLoss
        forward method. We provide this wrapper to expose the most recently
        computed loss (detatched from the current graph) to be used later by
        optimizers who require it (e.g., BWSGD).

        :param logits: the model logits
        :type logits: PyTorch FloatTensor objecet (n, c)
        :param y: the labels (or initial predictions) of the inputs (e.g., x)
        :type y: PyTorch Tensor object (n,)
        :return: the current loss
        :rtype: PyTorch FloatTensor (n,)
        """
        curr_loss = super().forward(logits, y)
        self.curr_loss = curr_loss.detatch()
        return curr_loss


class CWLoss(torch.nn.Module):
    """
    This class implements the Carlini-Wagner loss, as shown in
    https://arxiv.org/pdf/1608.04644.pdf. Specifically, the loss is defined as:

            lp(Δ) + c * max(max(f(x + Δ)_i:i ≠ y) - f(x + Δ)_y, -k)

    where Δ is the current perturbation vector to produce adversarial examples,
    x is the original input, y contains the labels of x, i is the next closest
    class (as measured through the model logits, returned by f), lp computes
    the p-norm of Δ, while k, the first hyperparameter, controls the minimum
    difference between the next closest class & the original class (i.e., how
    deep adversarial examples are pushed across the decision boundary), and c,
    the second hyperparameter, parameterizes the tradeoff between
    imperceptibility and misclassification.

    :func:`__init__`: instantiates a CWLoss object
    :func:`forward`: returns the loss for a given batch of inputs
    """

    del_req = True
    max_obj = False

    def __init__(self, norm=2, c=1, k=0):
        """
        This method instantiates a CWLoss object. It accepts three arguments:
        (1) norm, the lp-norm to use, (2) c, which emphasizes optimizing
        adversarial goals over the introduced distortion, and (3) k, which
        controls how far adversarial examples are pushed across the decision
        boundary (as measured through the difference between the initial and
        current predictions).

        :param norm: lp-space to project gradients into
        :type norm:
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
    model logits. Thus, when x is duplicated c times (where c is the number of
    classes) backpropogating the gradients from this loss function will return
    a model Jacobian. Moreover, like other losses, we store the last computed
    loss, and expose attributes to state whether optimizers should maximize
    this function and whether it requires a copy of the original input.

    :func:`__init__`: instantiates an IdentityLoss object
    :func:`forward`: returns the yth logit component for a batch of imputs
    """

    max_obj = False
    req_x = False

    def __init__(self):
        """
        This method instantiates an IdentityLoss object. It accepts no
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

        :param logits: the model logits
        :type logits: PyTorch FloatTensor objecet (n, c)
        :param y: the labels (or initial predictions) of the inputs (e.g., x)
        :type y: PyTorch Tensor object (n,)
        :return: the current loss
        :rtype: PyTorch FloatTensor (n,)
        """
        curr_loss = logits.take(y)
        self.curr_loss = curr_loss.detatch()
        return curr_loss


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
