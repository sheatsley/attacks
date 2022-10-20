"""
This module defines custom PyTorch-based optimizers.
Authors: Blaine Hoak & Ryan Sheatsley
Wed Jun 29 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
from torch.optim import Adam  # Implements Adam: A Method for Stochasitc Optimization
from torch.optim import SGD  # Implements stochasitc gradient descent

# TODO
# implement MomentumBestStart
# implmenet BackwardSGD


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
        Δ_(k+1) = P(Δ_k + α * (δ_(k+1) - Δ_k) + (1 - α) * (Δ_k - Δ_(k-1))

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
    found to be true, Δ is also reset to the vector that attained the highest
    value of L thus far. Conceptually, these optimizations augment PGD via
    momentum, adjust the learning rate to ensure adversarial goals are met
    continuously, and consume budget only if it aids in adversarial goals. As
    this optimizer requires computing the loss, the attibute loss_req is set to
    True.

    :func:`__init__`: instantiates MomentumBestStart objects
    :func:`step`: applies one optimization step
    """

    def __init__(
        self,
        params,
        atk_loss,
        epochs,
        epsilon,
        alpha=0.75,
        eta=0.75,
        pdecay=0.03,
        min_plen=0.06,
        **kwargs
    ):
        """
        This method instanties a MomentumBest Start object. It requires the
        total number of optimization iterations and a reference to a Loss
        object (so that the most recently computed loss can be retrieved). It
        also accepts keyword arguments to provide a homogeneous interface.
        Notably, because PyTorch optimizers cannot be instantiated without the
        parameter to optimize, a dummy tensor is supplied and expected to be
        overriden at a later time (i.e., within Traveler objects initialize
        method).

        :param params: the parameters to optimize over
        :type params: PyTorch FloatTensor object (n, m)
        :param atk_loss: returns the current loss of the attack
        :type atk_loss: Loss object
        :param epochs: total number of optimization iterations
        :type epochs: int
        :param epsilon: lp-norm threat model
        :type epsilon: float
        :param alpha: momentum factor
        :type alpha: float
        :param eta: minimum percentage of successful updates between checkpoints
        :type eta: float
        :param pdecay: period length decay
        :type pdecay: float
        :param min_plen: minimum period length
        :type min_plen: float
        :return: Momemtum Best Start optimizer
        :rtype: MomentumBestStart object
        """

        # precompute checkpoints
        pj = [0, 0.22]
        while pj[-1] < 1:
            pj.append(pj[-1] + max(pj[-1] - pj[-2] - pdecay, min_plen))
        super().__init__(
            [params],
            {
                "atk_loss": atk_loss,
                "alpha": alpha,
                "epochs": epochs,
                "epsilon": epsilon,
                "eta": eta,
                "checkpoints": {-int(-p * epochs) for p in pj[:-1]},
            },
        )

        # initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["best_p"] = p.detach().clone()
                state["momentum_buffer"] = p.detatch().clone()
                state["epoch"] = 0
                state["max_loss"] = 0
                state["pre_loss"] = 0
                state["lr"] = torch.full([p.size(0)], group["epsilon"])
                state["lr_updated"] = torch.full(state["lr"].size(), False)
                state["num_loss_updates"] = torch.full(state["lr"].size(), False)
                state["max_loss_updated"] = torch.full(state["lr"].size(), False)
                #state["step"] = 0
        return None

    @torch.no_grad()
    def step(self):
        """
        This method applies one optimization step as described above.
        Specifically, this optimizer: (1) steps in the direction of gradient of
        the loss with dynamic learning rate η, (2) applies a momentum step,
        parameterized by α, from the last two iterates, and (3) restarts the
        search locally and halves the learning rate if progress has stalled
        between checkpoints.

        :return: None
        :rtype: NoneType
        """
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad.data if self.maximize else -p.grad.data
                state = self.state[p]

                # update max loss and best perturbations, element-wise
                max_loss_inc = torch.gt(self.atk_loss.curr_loss, state["max_loss"])
                state["max_loss"][max_loss_inc] = self.atk_loss.curr_loss[max_loss_inc]
                state["best_p"][max_loss_inc] = p.detatch().clone()[max_loss_inc]

                # apply perturbation and momoentum steps
                state["momentum_buffer"].mul_(group["alpha"] - 1)
                grad.mul_(state["lr"].mul(group["alpha"]))
                p.mul_(2).sum_(group["momentum_buffer"]).sum_(grad)

                # perform checkpoint subroutines (and update associated info)
                loss_inc = torch.gt(self.atk_loss.curr, state["prev_loss"])
                state["num_loss_updates"][loss_inc].add_(1)
                state["max_loss_updated"][max_loss_inc] = True
                state["prev_loss"] = self.atk_loss.curr
                if state["step"] in group["checkpoints"]:

                    # has the loss increased >eta% of steps within this checkpoint?

                # update optimizer state

        return None
