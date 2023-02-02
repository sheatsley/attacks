"""
This module defines custom PyTorch-based optimizers.
Authors: Blaine Hoak & Ryan Sheatsley
Wed Jun 29 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
from torch.optim import Adam  # Implements Adam: A Method for Stochasitc Optimization
from torch.optim import SGD  # Implements stochasitc gradient descent


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
    this optimizer. Specifically, this optimizer performs both biased &
    backward steps towards the initial input. The biased step is defined as:

        Δ_(k+1) = (1 - α)(Δ_k + η * ∇f(x + Δ_k)) + α * η * P(∇f(x + Δ_k))

    where Δ is the current perturbation vector, k is the current iteration, η
    is the learnign rate, ∇f is the gradient of the model with respect to Δ, x
    is the initial input, P is a function designed to encourage perturbations
    to stay close to the initial input, and α is defined as:

     α = min(||∇f(x + Δ)||_p / (||∇f(x + Δ)||_p + ||P(∇f(x + Δ_k))||_p), α_max)

    where α_max is a constant in [0, 1]. Conceptually, this biased step
    encourages updates that are also close to the initial input. The biased
    step is defined as:

                        ß_(k+1) = ß if ∇f(x + Δ_k) ≠ c else 1
                        Δ_(k+1) = ß_(k+1) * η * ∇f(x + Δ_k)

    where k is the current iteration, ß is a constant in [0, 1], Δ is the
    current perturbation vector to produce adversarial example, ∇f is the
    gradient of the model with respect to Δ, x is the original input, c is the
    label, and η is the learning rate. Conceputally, ß enables adversaries to
    dampen the learning rate when inputs are already misclassified, Δ is as
    small as possible (while achieving adversarial goals).

    :func:`__init__`: instantiates BackwardSGD objects
    :func:`step`: applies one optimization step
    """

    def __init__(
        self,
        params,
        atk_loss,
        lr,
        maximize,
        norm,
        smap,
        alpha_max=0.1,
        beta=0.9,
        **kwargs,
    ):
        """
        This method instanties a Backward SGD object. Beyond standard optimizer
        parameters, a reference to a Loss module object (so misclassified
        inputs can be calculated), the lp-norm (to compute α), and a reference
        to a Saliency module object (as to retrieve P). It also accepts keyword
        arguments for to provide a homogeneous interface. Notably, because
        PyTorch optimizers cannot be instantiated without the parameter to
        optimize, a dummy tensor is supplied and expected to be overriden at a
        later time (i.e., within Traveler objects initialize method).

        :param alpha_max: maximum value of alpha
        :type alpha_max: float
        :param atk_loss: used to return the current model accuracy
        :type atk_loss: Loss object
        :param beta: backward step strength
        :type beta: float
        :param lr: learning rate
        :type lr: float
        :param maximize: whether to maximize or minimize the objective function
        :type maximize: bool
        :param minimum: minimum gradient value (to mitigate underflow)
        :type minimum: float
        :param params: the parameters to optimize over
        :type params: tuple of torch Tensor objects (n, m)
        :param smap: saliency map that computes P
        :type smap: surface module object
        :return: Backward SGD optimizer
        :rtype: BackwardSGD object
        """
        super().__init__(
            params,
            {
                "alpha_max": alpha_max,
                "atk_loss": atk_loss,
                "beta": beta,
                "lr": lr,
                "maximize": maximize,
                "norm": norm,
                "smap": smap,
            },
        )

    @torch.no_grad()
    def step(self, minimum=1e-8):
        """
        This method applies one optimization step as described above.
        Specifically, this optimizer: (1) performs a backward step for
        misclassified inputs parameterized by ß, and (2) steps in the direction
        of the gradient of the loss with static learning rate η. (2) computes a
        biased projection between the current gradient of the loss with static
        learning rate η and the original input (if provided), set by α.

        :param minimum: minimum gradient value (to mitigate underflow)
        :type minimum: float
        :return: None
        :rtype: NoneType
        """
        for group in self.param_groups:
            for p in group["params"]:
                lp = group["norm"]
                loss = group["atk_loss"]
                grad = p.grad.data
                p_grad = getattr(group["smap"], "org_proj", torch.zeros_like(grad))
                grad, p_grad = (grad, p_grad) if group["maximize"] else (-grad, -p_grad)

                # perform a backwardstep step for misclassified inputs
                misclassified = ~loss.acc
                p[misclassified] = p[misclassified].mul_(group["beta"])

                # compute alpha for biased projection
                grad_norm = grad.norm(lp, dim=1, keepdim=True)
                p_grad_norm = p_grad.norm(lp, dim=1, keepdim=True)
                norm_sum = p_grad_norm.add_(grad_norm).clamp_(minimum)
                alpha = grad_norm.div_(norm_sum).clamp_(max=group["alpha_max"])

                # apply biased projection and update step
                bias = p_grad.mul_(group["lr"]).mul_(alpha)
                grad.mul_(group["lr"]).mul_(1 - alpha).add_(bias)
                p.add_(grad)
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
    gradient of the model with respect to Δ, x is the initial input, and α is a
    constant in [0, 1]. Secondly, the learning rate η is potentially halved at
    every checkpoint w, where checkpoints are determined by multiplying the
    total number of attack iterations by p_j, defined as:

                p_(j+1) = p_j + max(p_j - p_(j-1) - 0.03, 0.06)

    where j is the current checkpoint and p_0 = 0 & p_1 = 0.22. At every
    checkpoint w_j, the following conditions are evaluated:

        (1) Σ_(i=w_(j-1))^(w_j - 1) 1 if L(x + Δ_(i+1)) > L(x + Δ_i)
                                    < ρ * (w_j - w_(j-1))
        (2) η_(w_(j-1)) == η_(w_j) and LMax_(w_(j-1)) == LMax_(w_j)

    where w_j is the jth checkpoint, L is the attack loss, x is the initial
    input, Δ_i is the perturbation vector at the ith iteration, ρ is a
    constant, η_(w_j) is the learning rate at checkpoint w_j, and LMax_(w_j) is
    the highest attack loss observed at checkpoint w_j. If both condtions are
    found to be true, Δ is also reset to the vector that attained the highest
    value of L thus far. Conceptually, these optimizations augment PGD via
    momentum, adjust the learning rate to ensure adversarial goals are met
    continuously (and consume budget only if it helps). As this optimizer
    requires computing the loss, the attibute loss_req is set to True.

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
        pdecay=0.03,
        pmin=0.06,
        rho=0.75,
        **kwargs,
    ):
        """
        This method instanties a MomentumBest Start object. Beyond standard
        optimizer parameters, it requires the total number of optimization
        iterations, a reference to a Loss module object (so that the most
        recently computed loss can be retrieved), and the maximimum budget of
        the lp-norm threat model (which initializes the learning rate). It also
        accepts keyword arguments to provide a homogeneous interface. Notably,
        because PyTorch optimizers cannot be instantiated without the parameter
        to optimize, a dummy tensor is supplied and expected to be overriden at
        a later time (i.e., within Traveler objects initialize method).

        :param atk_loss: used to return the current loss of the attack
        :type atk_loss: Loss object
        :param alpha: momentum factor
        :type alpha: float
        :param epochs: total number of optimization iterations
        :type epochs: int
        :param epsilon: lp-norm threat model budget
        :type epsilon: float
        :param pdecay: period length decay
        :type pdecay: float
        :param pmin: minimum period length
        :type pmin: float
        :param maximize: whether to maximize or minimize the objective function
        :type maximize: bool
        :param params: the parameters to optimize over
        :type params: tuple of torch Tensor objects (n, m)
        :param rho: minimum percentage of successful updates between checkpoints
        :type rho: float
        :return: Momemtum Best Start optimizer
        :rtype: MomentumBestStart object
        """

        # precompute checkpoints (first checkpoint is initialized to 22%)
        wdecay, wmin, wj = (max(int(epochs * w), 1) for w in (pdecay, pmin, 0.22))
        wj = [0, wj]
        while wj[-1] < epochs:
            wj.append(wj[-1] + max(wj[-1] - wj[-2] - wdecay, wmin))
        super().__init__(
            params,
            {
                "alpha": alpha,
                "atk_loss": atk_loss,
                "checkpoints": {w for w in wj[1:-1]},
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
                state["best_p"] = p.clone()
                state["best_g"] = torch.zeros_like(p)
                state["epoch"] = 0
                state["lr"] = torch.full((p.size(0), 1), 2 * group["epsilon"])
                state["lr_updated"] = torch.zeros(p.size(0), dtype=torch.bool)
                state["num_l_updates"] = torch.zeros(p.size(0), dtype=torch.int)
                state["best_l"] = torch.full(
                    (p.size(0),), -torch.inf if maximize else torch.inf
                )
                state["best_l_updated"] = torch.zeros(p.size(0), dtype=torch.bool)
                state["momentum_buffer"] = p.clone()
                state["prev_l"] = torch.zeros(p.size(0))
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
                inc = group["maximize"]
                loss = group["atk_loss"].loss
                grad = p.grad.data if group["maximize"] else -p.grad.data
                state = self.state[p]

                # save best loss with perturbations and gradients (saves a pass)
                best_l = loss.gt(state["best_l"]) if inc else loss.lt(state["best_l"])
                state["best_l"][best_l] = loss[best_l]
                state["best_p"][best_l] = p[best_l].clone()
                state["best_g"][best_l] = grad[best_l].clone()

                # perform checkpoint subroutines (and update associated info)
                loss_inc = loss.gt(state["prev_l"]) if inc else loss.lt(state["prev_l"])
                state["num_l_updates"][loss_inc] += 1
                state["best_l_updated"][best_l] = True
                state["prev_l"] = loss
                if state["epoch"] in group["checkpoints"]:

                    # loss improved <rho% or lr and best loss stayed the same?
                    c1 = state["num_l_updates"].lt(group["rho"] * state["step"])
                    c2 = ~(state["lr_updated"].logical_or(state["best_l_updated"]))
                    update = c1.logical_or(c2)

                    # if so, half learning rate and reset perturbation to best loss
                    state["lr"][update] /= 2
                    p[update] = state["best_p"][update].clone()
                    grad[update] = state["best_g"][update].clone()

                    # update checkpoint subroutine state
                    state["step"] = 0
                    state["lr_updated"][update] = True
                    state["lr_updated"][~update] = False
                    state["num_l_updates"].fill_(0)
                    state["best_l_updated"].fill_(False)

                # apply update step (simplified to mitigate underflow)
                momentum = state["momentum_buffer"].mul_(group["alpha"] - 1)
                state["momentum_buffer"] = p.clone()
                grad.mul_(state["lr"]).mul_(group["alpha"])
                p.mul_(2 - group["alpha"]).add_(grad).add_(momentum)

                # update optimizer state
                state["step"] += 1
                state["epoch"] += 1
        return None
