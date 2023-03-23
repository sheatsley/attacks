"""
This module defines custom PyTorch-based loss functions referenced in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Thu Feb 2 2023
"""
import torch


class CELoss(torch.nn.CrossEntropyLoss):
    """
    This class is identical to the CrossEntropyLoss class in PyTorch, with the
    exception that: (1) the most recently computed loss is stored in curr_loss
    (to faciliate optimizers who require it) and (2) two additional attributes
    are added: p_req (set to False) and max_obj (set as True). We add these
    attributes to state that optimizers should maximize this function and that
    a reference to perturbation vectors is not needed by this class.

    :func:`__init__`: instantiates a CELoss object
    :func:`forward`: returns the loss for a given batch of inputs
    """

    p_req = False
    max_obj = True

    def __init__(self, **_):
        """
        This method instantiates a CELoss object. It accepts keyword arguments
        for the PyTorch parent class described in:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        We note that we explicitly pass reduction="none", as this can cause
        gradients to underflow when computing adversarial examples for large
        batch sizes.

        :return: Cross Entropy loss
        :rtype: CELoss object
        """
        super().__init__(reduction="none")
        return None

    def forward(self, logits, y, yt=None):
        """
        This method serves as a wrapper for the PyTorch CrossEntropyLoss
        forward method. We provide this wrapper to expose the most recently
        computed accuracy and loss to be used later by optimizers who require
        it (e.g., BackwardSGD & MomentumBestStart, respectively).

        :param logits: the model logits
        :type logits: torch Tensor object (n, c)
        :param y: the labels (or initial predictions) of the inputs (e.g., x)
        :type y: torch Tensor object (n,)
        :param yt: the true labels if attempting to compute a jacobian
        :type yt: torch Tensor object (n,)
        :return: the current loss
        :rtype: torch Tensor object (n,)
        """
        loss = super().forward(logits, y)
        if loss.requires_grad:
            self.loss, self.acc = record(loss, logits, y, y if yt is None else yt)
        return loss


class CWLoss(torch.nn.Module):
    """
    This class implements the Carlini-Wagner loss, as shown in
    https://arxiv.org/pdf/1608.04644.pdf. Specifically, the loss is defined as:

            ||Δ||_2 + c * max(max(f(x + Δ)_i:i ≠ y) - f(x + Δ)_y, -k)

    where Δ is the current perturbation vector to produce adversarial examples,
    x is the original input, y contains the labels of x, i is the next closest
    class (as measured through the model logits, returned by f), ||·|| computes
    the l2-norm of Δ, while k, the first hyperparameter, controls the minimum
    difference between the next closest class & the original class (i.e., how
    deep adversarial examples are pushed across the decision boundary), and c,
    the second hyperparameter, parameterizes the tradeoff between
    imperceptibility and misclassification (which can be dynamically optimized
    for, e.g., via binary search). Moreover, like other losses, we store the
    last computed accuracy & loss, and expose attributes to state that
    optimizers should minimize this function and Surface objects should provide
    a reference to the perturbation vector.

    :func:`__init__`: instantiates a CWLoss object
    :func:`attach`: sets the perturbation vector as an object attribute
    :func:`forward`: returns the loss for a given batch of inputs
    """

    p_req = True
    max_obj = False

    def __init__(self, classes, c=1.0, k=0.0):
        """
        This method instantiates a CWLoss object. It accepts four arguments:
        (1) the number of classes (so that logit differences can be computed
        accurately) (2) norm, the lp-norm to use, (3) c, which emphasizes
        optimizing adversarial goals over the introduced distortion, and (4) k,
        which controls how far adversarial examples are pushed across the
        decision boundary (as measured through the logit differences of the yth
        logit and the next largest logit). Finally, a reference to c is saved
        and exposed as an optimizable hyperparameter.

        :param num_classses: number of classes
        :type classes: int
        :param norm: lp-space to project gradients into
        :type norm: supported ord arguments in torch linalg.vector_norm function
        :param c: importance of misclassification over imperceptability
        :type c: float
        :param k: minimum logit difference between true and current classes
        :type k: float
        :return: Carlini-Wagner loss
        :rtype: CWLoss object
        """
        super().__init__()
        self.classes = classes
        self.c = torch.tensor((c,))
        self.k = k
        self.hparam = ("c", self.c)
        return None

    def attach(self, p):
        """
        This method serves as a setter for attaching a reference to the
        perturbation vector to CWLoss objects as an attribute, as the attribute
        is subsequently referenced in the forward pass. If necessary, c is
        expanded in-place by the number of inputs to faciliate hyperparameter
        optimization, as done in https://arxiv.org/pdf/1608.04644.pdf.

        :param p: perturbation vector
        :type p: torch Tensor object (n, m)
        :param x: initial inputs
        :type x: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """
        self.p = p
        if self.c.size(0) < p.size(0):
            self.c.resize_(p.size(0)).fill_(self.c[0])
        return None

    def forward(self, logits, y, yt=None):
        """
        This method computes the loss described above. Specifically, it
        computes the sum of: (1) the lp-norm of the perturbation, and (2) the
        logit difference between the true class and the maximum logit not equal
        to the true class. Moreover, if attack parameters require a saliency
        map (detected via a shape mismatch between the pertrubation vector and
        the model logits), the l2-norm vector is expanded by a factor c (where
        c is the number of classes).

        :param logits: the model logits
        :type logits: torch Tensor object (n, c)
        :param y: the labels (or initial predictions) of the inputs (e.g., x)
        :type y: torch Tensor object (n,)
        :param yt: the true labels if attempting to compute a jacobian
        :type yt: torch Tensor object (n,)
        :return: the current loss
        :rtype: torch Tensor object (n,)
        """

        # compute l2-norm of perturbation vector
        l2 = self.p.norm(dim=1).repeat_interleave(logits.size(0) // self.p.size(0))

        # compute logit differences
        y_hot = torch.nn.functional.one_hot(y, num_classes=self.classes).bool()
        yth_logit = logits.masked_select(y_hot)
        max_logit, _ = logits.masked_select(~y_hot).view(-1, logits.size(1) - 1).max(1)
        log_diff = torch.clamp(yth_logit.sub(max_logit), min=-self.k)

        # save current accuracy and loss for optimizers later
        c = self.c.repeat_interleave(logits.size(0) // self.p.size(0))
        loss = c.mul(log_diff).add(l2)
        if loss.requires_grad:
            self.loss, self.acc = record(loss, logits, y, y if yt is None else yt)
        return loss


class DLRLoss(torch.nn.Module):
    """
    This class implements the Difference of Logits Ratio loss function, as
    shown in https://arxiv.org/pdf/2003.01690.pdf. Specifically, the loss
    is defined as:

            -((f(x + Δ)_y - max(f(x + Δ)_i:i ≠ y)) / (π_1 - π_3))

    where f returns the model logits, x is the original input, Δ is the current
    perturbation vector to produce adversarial examples, y is the true class, i
    is the next closest class (as measured by the moodel logits), and π defines
    the descending order of the model logits (that is, the denominator
    represents the difference between the largest and 3rd largest model logit).
    Like other losses, we store the last computed accuracy & loss, and expose
    attributes so that optimizers maximize this function and that a reference
    to the perturbation vector is not needed.

    :func:`__init__`: instantiates a DLRLoss  object
    :func:`forward`: returns the loss for a given batch of inputs
    """

    p_req = False
    max_obj = True

    def __init__(self, classes, minimum=1e-8):
        """
        This method instantiates an IdentityLoss object. It accepts the number
        of classes (so that logit differences can be computed accurately).
        Notably, the denominator in the loss above is set to 1 for binary
        classification.

        :param classses: number of classes
        :type classes: int
        :param minimum: minimum gradient value (to mitigate underflow)
        :type minimum: float
        :return: Identity loss
        :rtype: IdentityLoss object
        """
        super().__init__()
        self.classes = classes
        self.minimum = minimum
        self.d = (
            lambda x: x[:, 0].sub(x[:, 2])
            if classes > 2
            else torch.tensor(1.0, device=x.device())
        )
        return None

    def forward(self, logits, y, yt=None):
        """
        This method computes the loss described above. Specifically, it
        computes the division of: (1) the logit difference between the yth
        logit and largest non-yth logit, and (2) the difference between the
        largest logit and 3rd largest logit (when the number of classes is
        greater than three, otherwise 1 is returned).

        :param logits: the model logits
        :type logits: torch Tensor object (n, c)
        :param y: the labels (or initial predictions) of the inputs (e.g., x)
        :type y: torch Tensor object (n,)
        :param yt: the true labels if attempting to compute a jacobian
        :type yt: torch Tensor object (n,)
        :return: the current loss
        :rtype: torch Tensor object (n,)
        """

        # compute logit differences
        y_hot = torch.nn.functional.one_hot(y, num_classes=self.classes).bool()
        yth_logit = logits.masked_select(y_hot)
        max_logit, _ = logits.masked_select(~y_hot).view(-1, logits.size(1) - 1).max(1)
        log_diff = yth_logit.sub(max_logit)

        # compute ordered logit differences
        log_desc = logits.sort(dim=1, descending=True).values
        pi_diff = self.d(log_desc)
        loss = -(log_diff.div(pi_diff.clamp_(self.minimum)))
        if loss.requires_grad:
            self.loss, self.acc = record(loss, logits, y, y if yt is None else yt)
        return loss


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
    accuracy & loss, and expose attributes so that optimizers should minimize
    this function and a reference to the perturbation vector is not needed.

    :func:`__init__`: instantiates an IdentityLoss object
    :func:`forward`: returns the yth logit component for a batch of imputs
    """

    p_req = False
    max_obj = False

    def __init__(self, **_):
        """
        This method instantiates an IdentityLoss object. It accepts no
        arguments.

        :return: Identity loss
        :rtype: IdentityLoss object
        """
        super().__init__()
        return None

    def forward(self, logits, y, yt=None):
        """
        This method computes the loss described above. Specifically, it returns
        the yth-component of the logits.

        :param logits: the model logits
        :type logits: torch Tensor object (n, c)
        :param y: the labels (or initial predictions) of the inputs (e.g., x)
        :type y: torch Tensor object (n,)
        :param yt: the true labels if attempting to compute a jacobian
        :type yt: torch Tensor object (n,)
        :return: the current loss
        :rtype: torch Tensor object (n,)
        """
        loss = logits.gather(1, y.unsqueeze(1)).flatten()
        if loss.requires_grad:
            self.loss, self.acc = record(loss, logits, y, y if yt is None else yt)
        return loss


def record(loss, logits, y, yt):
    """
    This function computes the accuracy and loss (to be used by optimizers that
    require it, e.g., BackwardSGD and MomentumBestStart). Specifically, as
    inputs are duplicated class-number of times for attacks that require model
    Jacobians, we index into the loss and logits corresponding to the true
    labels, given by yt (which is correct when yt == y for other attacks).

    :param logits: the model logits
    :type logits: torch Tensor object (n, c)
    :param y: the labels (or initial predictions) of the inputs (e.g., x)
    :type y: torch Tensor object (n,)
    :param yt: the true labels if attempting to compute a jacobian
    :type yt: torch Tensor object (n,)
    """
    classes = y.numel() // yt.numel()
    offset = yt if y.numel() != yt.numel() else 0
    loss = loss.detach().take(
        torch.arange(0, y.numel(), classes, device=loss.device).add(offset)
    )
    accuracy = logits.detach()[::classes].argmax(1).eq(yt)
    return loss, accuracy
