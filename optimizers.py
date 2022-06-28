import torch
from torch import Tensor
from typing import List, Optional
import math


class SGD(torch.optim.SGD):
    """
    This class provides an alias for the pytorch implementation of SGD
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
        **kwargs
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )


class Adam(torch.optim.Adam):
    """
    This class provides an alias for the pytorch implementation of SGD
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )


class MBS(torch.optim.Optimizer):
    """
    This class defines the optimizer used in the AutoPGD attack. Specifically, it
    implements SGD with momentum but introduces a best start component that also updates
    the learning rate.
    """

    def __init__(
        self,
        params,
        lr,
        compute_loss,
        n_iter=1000,
        alpha=0.75,
        maximize=False,
        **kwargs
    ) -> None:
        """
        :params: parameters to update
        :lr: learning rate
        :alpha: previous update influence regulation parameter
        :maximize: whether to maximize or minimize the objective
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, alpha=alpha, maximize=maximize, n_iter=n_iter)
        super().__init__(params, defaults)
        self.compute_loss = compute_loss
        self.n_iter = n_iter

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("alpha", 0.75)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # check if we still need to initialize the best start
                if len(state) == 0:
                    state["momentum_buffer"] = p
                    state["step"] = 0
                    state["learning_rate"] = group["lr"] * torch.ones(p.shape[0])
                    with torch.no_grad():
                        state["fmax"] = self.compute_loss(p)
                        state["xmax"] = p
                    state["prev_loss"] = state["fmax"]
                    # checks if learning rate has been updated since the last checkpoint
                    state["lr_updated_cp"] = torch.zeros(p.shape[0])
                    # checks if fmax has been updated since the last checkpoint
                    state["fmax_updated_cp"] = torch.zeros(p.shape[0])
                    # counts the number of times f has increased since the last checkpoint
                    state["f_inc_cp_count"] = torch.zeros(p.shape[0])
                    state["curr_cp"] = 0
                    state["next_cp"] = 0.22

                state["step"] += 1

                if state["step"] == 1:
                    p.sub_(state["learning_rate"].unsqueeze(1) * grad)
                else:
                    z_k = p - state["learning_rate"].unsqueeze(1) * grad
                    temp_buf = p
                    p.add_(
                        group["alpha"] * (z_k - p)
                        + ((1 - group["alpha"]) * (p - state["momentum_buffer"]))
                    )
                    state["momentum_buffer"] = temp_buf
                # compute update xmax and fmax element wise
                with torch.no_grad():
                    loss_stack = torch.stack((state["fmax"], self.compute_loss(p)))
                idx = loss_stack.argmax(dim=0, keepdim=True)
                # if idx is 1, then fmax is being updated
                state["fmax_updated_cp"] += idx.squeeze(0)
                state["fmax"] = loss_stack.gather(dim=0, index=idx).squeeze(0)
                state["xmax"] = (
                    torch.stack((state["xmax"], p))
                    .gather(
                        dim=0,
                        index=idx.squeeze(0)
                        .unsqueeze(1)
                        .repeat(1, p.shape[1])
                        .unsqueeze(0),
                    )
                    .squeeze(0)
                )
                # f is increasing if the current loss is greater than the previous loss
                state["f_inc_cp_count"] += torch.gt(loss_stack[1], state["prev_loss"])
                state["prev_loss"] = loss_stack[1]
                # check if we are at a checkpoint
                if state["step"] == math.ceil(state["next_cp"] * self.n_iter):
                    # see if less than 75% of iters in this checkpoint have increased loss
                    cond1 = torch.gt(
                        state["f_inc_cp_count"],
                        0.75
                        * (state["step"] - math.ceil(state["curr_cp"] * self.n_iter)),
                    )
                    # check if the learning rate and fmax have been updated less than one time
                    cond2 = torch.lt(
                        state["lr_updated_cp"],
                        torch.ones_like(state["lr_updated_cp"]),
                    ) * torch.lt(
                        state["fmax_updated_cp"],
                        torch.ones_like(state["fmax_updated_cp"]),
                    )
                    # for condtrue, update lr, xmax, and lr_updated_cp
                    condtrue = cond1 * cond2
                    # divide lr by 2 if the condition is true, 1 otherwise (keep same lr)
                    state["learning_rate"] /= condtrue + 1
                    # update xmax if the condition is true, else keep it the same
                    state["xmax"] = (
                        state["xmax"] * torch.logical_not(condtrue).unsqueeze(1)
                    ) + (p * condtrue.unsqueeze(1))
                    # if condtrue, we are updating the learning rate in this checkpoint
                    state["lr_updated_cp"] = condtrue

                    # always recalculate next_cp and update curr_cp, reset all cp based
                    # values
                    state["f_inc_cp_count"] = torch.zeros(p.shape[0])
                    state["fmax_updated_cp"] = torch.zeros(p.shape[0])
                    tmp_cp = state["next_cp"]
                    # compute new checkpoint
                    state["next_cp"] += max(
                        state["next_cp"] - state["curr_cp"] - 0.03, 0.06
                    )
                    # update prev checkpoint
                    state["curr_cp"] = tmp_cp
        return


class BWSGD(torch.optim.Optimizer):
    """
    This class defines the optimizer used in the AutoPGD attack. Specifically, it
    implements SGD with momentum but introduces a best start component that also updates
    the learning rate.
    """

    def __init__(
        self,
        params,
        lr,
        compute_misclass,
        alpha_max=0.0,
        beta=0.9,
        maximize=False,
        **kwargs
    ) -> None:
        """
        :params: parameters to update
        :lr: learning rate
        :alpha: previous update influence regulation parameter
        :maximize: whether to maximize or minimize the objective
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, alpha_max=alpha_max, maximize=maximize, beta=beta)
        super().__init__(params, defaults)
        self.compute_is_misclass = compute_misclass

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("alpha_max", 0.0)
            group.setdefault("beta", 0.9)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # check if we still need to initialize the best start
                if len(state) == 0:
                    state["step"] = 0
                    state["p_og"] = torch.clone(p)

                state["step"] += 1

                if group["alpha_max"] == 0.0:
                    p.sub_(group["lr"] * grad)
                else:
                    raise (NotImplementedError)

                p.mul_(group["beta"]).add_(state["p_og"] * (1 - group["beta"]))
                # if we do bw step and still misclassify, then take step, else revert
                # back to original p before bw step
                p = torch.where(
                    self.compute_is_misclass(p).unsqueeze(1),
                    p,
                    (p - state["p_og"] * (1 - group["beta"])) / group["beta"],
                )
        return
