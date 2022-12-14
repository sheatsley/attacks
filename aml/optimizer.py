"""
This module defines custom PyTorch-based optimizers.
Authors: Blaine Hoak & Ryan Sheatsley
Wed Jun 29 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
from torch.optim import Adam  # Implements Adam: A Method for Stochasitc Optimization
from torch.optim import SGD  # Implements stochasitc gradient descent

# TODO
# consider adding closure to BackwardSGD to perform beta backstep after perturbation
# add support when maximize is false (eg MBS max loss cannot be initialized to zero)
# add unit tests to confirm correctness
# find better way to update p in BackwardSGD


class BackwardSGD(torch.optim.Optimizer):
    """
    This class implements BackwardSGD (as named by
    https://arxiv.org/pdf/2209.04521.pdf), an optimizer used in the Fast
    Adaptive Boundary Attack (FAB), as shown in
    https://arxiv.org/abs/1907.02044. FAB aims to to address two key
    deficiencies in PGD (as introduced in
    https://arxiv.org/pdf/1706.06083.pdf): (1) inability to find minimum norm
    adversarial examples (agnostic of lp budget), and (2) slow robust accuracy
    assessment. (2) is broadly addressed through the saliency map used by
    DeepFool (https://arxiv.org/pdf/1511.04599.pdf), while (1) is addressed by
    this optimizer. Specifically, this optimizer performs a biased backward
    step towards the original input, defined as:

                        ß_(k+1) = ß if ∇f(x + Δ_k) ≠ c else 1
                        Δ_(k+1) = ß_(k+1) * η * ∇f(x + Δ_k)

    where k is the current iteration, ß is a consant in [0, 1], Δ is the
    current perturbation vector to produce adversarial example, ∇f is the
    gradient of the model with respect to Δ, x is the original input, c is the
    label, and η is the learning rate. Conceputally, ß enables adversaires to
    dampen the learning rate when the inputs is already misclassified, ensuring
    x + Δ is as close to x as possible (while achieving adversarial goals).

    :func:`__init__`: instantiates BackwardSGD objects
    :func:`step`: applies one optimization step
    """

    def __init__(self, params, lr, maximize, model, beta=0.9, **kwargs):
        """
        This method instanties a Backward SGD object. It requires a learning
        rate, ß, and a reference to a dlm LinearClassifier-inherited object (as
        to determine misclassified inputs). It also accepts keyword arguments
        for to provide a homogeneous interface. Notably, because PyTorch
        optimizers cannot be instantiated without the parameter to optimize, a
        dummy tensor is supplied and expected to be overriden at a later time
        (i.e., within Traveler objects initialize method).

        :param beta: momentum factor
        :type beta: float
        :param lr: learning rate
        :type lr: float
        :param maximize: whether to maximize or minimize the objective function
        :type maximize: bool
        :param model: returns misclassified inputs
        :type model: dlm LinearClassifier-inherited object
        :param params: the parameters to optimize over
        :type params: tuple of torch Tensor objects (n, m)
        :return: Backward SGD optimizer
        :rtype: BackwardSGD object
        """
        super().__init__(
            params,
            {"lr": lr, "beta": beta, "maximize": maximize, "model": model},
        )

        # initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["beta"] = torch.full((p.size(0), 1), group["beta"])

    @torch.no_grad()
    def step(self):
        """
        This method applies one optimization step as described above.
        Specifically, this optimizer: (1) sets ß based on whether samples are
        misclassified, and (2) steps in the direction of the gradient of the
        loss with static learning rate η.

        :return: None
        :rtype: NoneType
        """
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad.data if group["maximize"] else -p.grad.data
                state = self.state[p]

                # set beta for misclassified inputs and apply update
                misclassified = ~group["model"].correct.unsqueeze(1)
                state["beta"] = torch.where(misclassified, group["beta"], 1)
                p[:] = grad.mul_(state["beta"].mul_(group["lr"]))
        return None


class MomentumBestStart(torch.optim.Optimizer):
    """
    This class implements MomentumBestStart (as named by
    https://arxiv.org/pdf/2209.04521.pdf), an optimizer used in the AutoPGD
    attack, as shown in https://arxiv.org/pdf/2003.01690.pdf. AutoPGD aims to
    address three key deficiencies in PGD (as introduced in
    https://arxiv.org/pdf/1706.06083.pdf): (1) a fixed learning rate, (2)
    obliviousness to budget, and (3) a lack of any measurement of progress;
    this optimizer addresses these weaknesses. Firstly, this optimizer adds a
    momenmtum term, defined as:

        δ_(k+1) = Δ_k + η * ∇f(x + Δ_k)
        Δ_(k+1) = Δ_k + α * (δ_(k+1) - Δ_k) + (1 - α) * (Δ_k - Δ_(k-1)

    where k is the current iteration, Δ is the current perturbation vector to
    produce adversarial examples, η is the current learning rate, ∇f is the
    gradient of the model with respect to Δ, x is the original input, and α is
    a constant in [0, 1]. Secondly, the learning rate η is potentially halved
    at every checkpoint w, where checkpoints are determined by multiplying the
    total number of attack iterations by p_j, defined as:

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
        maximize,
        alpha=0.75,
        min_plen=0.06,
        pdecay=0.03,
        rho=0.75,
        **kwargs
    ):
        """
        This method instanties a MomentumBest Start object. It requires the
        total number of optimization iterations, a reference to a Loss object
        (so that the most recently computed loss can be retrieved), and the
        maximimum budget of the lp-norm threat model (which initializes the
        learning rate). It also accepts keyword arguments to provide a
        homogeneous interface. Notably, because PyTorch optimizers cannot be
        instantiated without the parameter to optimize, a dummy tensor is
        supplied and expected to be overriden at a later time (i.e., within
        Traveler objects initialize method).

        :param atk_loss: returns the current loss of the attack
        :type atk_loss: Loss object
        :param alpha: momentum factor
        :type alpha: float
        :param epochs: total number of optimization iterations
        :type epochs: int
        :param epsilon: lp-norm threat model budget
        :type epsilon: float
        :param pdecay: period length decay
        :type pdecay: float
        :param maximize: whether to maximize or minimize the objective function
        :type maximize: bool
        :param min_plen: minimum period length
        :type min_plen: float
        :param params: the parameters to optimize over
        :type params: tuple of torch Tensor objects (n, m)
        :param rho: minimum percentage of successful updates between checkpoints
        :type rho: float
        :return: Momemtum Best Start optimizer
        :rtype: MomentumBestStart object
        """

        # precompute checkpoints
        pj = [0, 0.22]
        while pj[-1] < 1:
            pj.append(pj[-1] + max(pj[-1] - pj[-2] - pdecay, min_plen))
        super().__init__(
            params,
            {
                "atk_loss": atk_loss,
                "alpha": alpha,
                "checkpoints": {-int(-p * epochs) for p in pj[:-1]},
                "epochs": epochs,
                "epsilon": epsilon,
                "maximize": maximize,
                "rho": rho,
            },
        )

        # initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["best_p"] = p.detach().clone()
                state["epoch"] = 0
                state["lr"] = torch.full((p.size(0),), group["epsilon"])
                state["lr_updated"] = torch.full(state["lr"].size(), False)
                state["num_loss_updates"] = torch.zeros(state["lr"].size())
                state["max_loss"] = torch.zeros(state["lr"].size())
                state["max_loss_updated"] = torch.full(state["lr"].size(), False)
                state["momentum_buffer"] = p.detach().clone()
                state["prev_loss"] = torch.zeros(state["lr"].size())
                state["step"] = 0
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
                curr_loss = group["atk_loss"].curr_loss
                grad = p.grad.data if group["maximize"] else -p.grad.data
                state = self.state[p]

                # update max loss and best perturbations, element-wise
                max_loss_inc = curr_loss.gt(state["max_loss"])
                state["max_loss"][max_loss_inc] = curr_loss[max_loss_inc]
                state["best_p"][max_loss_inc] = p.detach().clone()[max_loss_inc]

                # apply perturbation and momoentum steps
                state["momentum_buffer"].mul_(group["alpha"] - 1)
                grad.mul_(state["lr"].mul(group["alpha"]).unsqueeze(1))
                p.mul_(2).add_(state["momentum_buffer"]).add_(grad)

                # perform checkpoint subroutines (and update associated info)
                loss_inc = curr_loss.gt(state["prev_loss"])
                state["num_loss_updates"][loss_inc].add_(1)
                state["max_loss_updated"][max_loss_inc] = True
                state["prev_loss"] = curr_loss
                if state["epoch"] in group["checkpoints"]:

                    # loss increased <rho% or lr and max loss stayed the same?
                    c1 = state["num_loss_updates"].gt(group["rho"] * state["step"])
                    c2 = state["lr_updated"].logical_and(state["max_loss_updated"])
                    update = c1.logical_or(c2)

                    # if so, half learning rate and reset perturbation to max loss
                    state["lr"][update] = state["lr"].div(2)[update]
                    p[update] = state["best_p"][update]

                    # update checkpoint subroutine state
                    state["step"] = 0
                    state["lr_updated"].mul_(False)
                    state["num_loss_updates"].mul_(0)
                    state["max_loss_updated"].mul_(False)

                # update optimizer state
                state["momentum_buffer"] = p.detach().clone()
                state["epoch"] += 1
                state["step"] += 1
        return None


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
