"""
This module defines custom Pytorch-based optimizers.
Authors: Blaine Hoak & Ryan Sheatsley
Wed Jun 29 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# implement MomentumBestStart
# implmenet BackwardSGD


class Adam(torch.optim.Adam):
    """
    This class is identical to the Adam class in PyTorch, with the exception
    that two additional attributes are added: req_loss (set to False) and
    req_acc (set to False). Unlike Adam, some optimizers (e.g.,
    MomentumBestStart & BackwardSGD) require the adversarial loss or model
    accuracy to perform their update step, and thus, we instantiate these
    parameters for a homogenous interface.

    :func:`__init__`: instantiates Adam objects
    """

    acc_req = False
    loss_req = False

    def __init__(self, atk_loss=None, model_acc=None, **kwargs):
        """
        This method instantiates an Adam object. It accepts keyword arguments
        for the PyTorch parent class described in:
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html.

        :param kwargs: keyword arguments for torch.optim.Adam
        :type kwargs: dict
        :return Adam optimizer
        :rtype: Adam object
        """
        super().__init__(**kwargs)
        return None


class MomentumBestStart(torch.optim.Optimizer):
    """
    This class implements MomentumBestStart (as named by [paper_url]), an
    optimizer used in the AutoPGD attack, as shown in
    https://arxiv.org/pdf/2003.01690.pdf. AutoPGD aims to address three key
    deficiencies in PGD (as introduced in
    https://arxiv.org/pdf/1706.06083.pdf): (1) a fixed learning rate, (2)
    obliviousness to budget, and (3) a lack of any measurement of progress;
    this optimizer addresses these weaknesses. Firstly, this optimizer adds a
    momenmtum term, defined as:

        δ_(k+1) = P(Δ_k + η * ∇f(x + Δ_k))
        Δ_(k+1) = P(Δ + α * (δ_(k+1) - Δ_k) + (1 - α) * (Δ_k - Δ_(k-1))

    where k is the current iteration, P projects inputs into the feasible set
    (i.e., an lp-norm), Δ is the current perturbation vector to produce
    adversarial examples, η is the current learning rate, ∇f is the gradient of
    the model with respect to Δ, x is the original input, and α is a constant
    in [0, 1]. Secondly, the learning rate η is potentially halved at every
    checkpoint w, where checkpoints are determined by multiplying the total
    number of attack iterations by p_j, defined as:

                p_(j+1) = p_j + max(p_j - p_(j-1) - 0.03, 0.06)

    where j is the current checkpoint and p_0 = 0 & p_1 = 0.22. At every
    checkpoint w_j, the following conditions are evaluated:

        (1) Σ_(i=w_(j-1))^(w_j - 1) 1 if L(x + Δ_(i+1)) > L(x + Δ_i)
                                    < ρ * (w_j - w_(j-1))
        (2) η_(w_(j-1)) == η_(w_j) and LMax_(w_(j-1)) == LMax_(w_j)

    where w_j is the jth checkpoint, L is the model loss, x is the original
    input, Δ_i is the perturbation vector at the ith iteration, ρ is a
    constant, η_(w_j) is the learning rate at checkpoint w_j, and LMax_(w_j) is
    the highest model loss observed at checkpoint w_j. If both condtions are
    found to be true, Δ is also reset to vector that attained the highest value
    of L thus far. Conceptually, these optimizations augment PGD via momentum,
    adjust the learning rate to ensure adversarial goals are met continuously,
    and consume budget only if it aids in adversarial goals. As this optimizer
    requires computing the loss, the attibute loss_req is set to True.

    :func:`__init__`: instantiates SGD objects
    """

    acc_req = False
    loss_req = True


class SGD(torch.optim.SGD):
    """
    This class is identical to the SGD class in PyTorch, with the exception
    that two additional attributes are added: req_loss (set to False) and
    req_acc (set to False). Unlike SGD, some optimizers (e.g.,
    MomentumBestStart & BackwardSGD) can require the adversarial loss or model
    accuracy to perform their update step, and thus, we instantiate these
    parameters for a homogenous interface.

    :func:`__init__`: instantiates SGD objects
    """

    acc_req = False
    loss_req = False

    def __init__(self, atk_loss=None, model_acc=None, **kwargs):
        """
        This method instantiates an SGD object. It accepts keyword arguments
        for the PyTorch parent class described in:
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.

        :param atk_loss: returns the current loss of the attack
        :type atk_loss: Loss object
        :param model_acc: returns the current model accuracy
        :type model_acc: ScikitTorch Model instance method
        :param kwargs: keyword arguments for torch.optim.SGD
        :type kwargs: dict
        :return Stochastic Gradient Descent optimizer
        :rtype: SGD object
        """
        super().__init__(**kwargs)
        return None