"""
This module defines the travler class referenced in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Wed Jul 21 2021
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add unit tests
# add Line Search


class Traveler:
    """
    This traveler class serves as a wrapper to support custom and COTS
    PyTorch-based optimizers for arbitrary surfaces (as described in the
    surface module). The methods defined within the class are desinged to
    facilitate crafting adversarial examples.

    :func:`__init__`: instantiates Traveler objects
    :func:`craft`: returns a batch of adversarial examples
    :func:`tanh_gradient`: returns the gradient of tanh-mapped tensors
    :func:`tanh_map`: maps a tensor into (and out of) tanh-space
    """

    def __init__(
        self,
        epochs,
        optimizer,
        alpha=0.01,
        random_alpha=0,
        change_of_variables=False,
        x_range=(0, 1),
    ):
        """
        This method instantiates a traveler object with a variety of attributes
        necessary for the remaining methods in this class. Specifically, the
        following attributes are collected: (1) the number of optimizer steps
        before returning, (2) a PyTorch Optimizer object, (3) if necessary, the
        learning rate applied to the opitmizer (i.e., alpha), (4) the magnitude
        of a random perturbation to the input before crafting, (5) whether the
        input should be transformed into the tanh-space, and (6) the
        permissible feature ranges. Moreover, to avoid mapping features to
        negative or positive infinity (i.e., when mapping to tanh-space),
        features are mapped between (0, 1) (non-inclusive) via the
        tanh_smoother attribute. Finally, an initialization flag is set to
        inform attack objects preprocessing steps have been completed when
        repeatedly calling craft.

        :param epochs: number of optimization steps to perform
        :type epochs: integer
        :param optimizer: optimization algorithm to use
        :type optimizer: PyTorch Optimizer class
        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param random_alpha: magnitude of a random perturbation
        :type random_alpha: scalar
        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: boolean
        :param x_range: specifies minimum and maximum feature values
        :type x_range: tuple of floats
        :return: a traveler
        :rtype: Traveler object
        """
        self.epochs = epochs
        self.optimizer = optimizer
        self.alpha = alpha
        self.random_alpha = random_alpha
        self.change_of_variables = change_of_variables
        self.x_range = x_range
        self.tanh_smoother = 0.999999
        self.init_req = True
        return None

    def craft(self, x, y, surface):
        """
        This method crafts adversarial examples by applying epochs number of
        optimization steps to x. The gradient information used by optimizers is
        made accessible by Surface objects.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: n x m tensor
        :param y: the labels (or initial predictions) of x
        :type y: n-length vector
        :param surface: surface containing the parameters to optimize over
        :type surface: Surface object
        :param x_min: minimum value for x
        :type x_min: scalar or n x m tensor
        :param x_max: maximum value for x
        :type x_max: scalar or n x m tensor
        :return: a batch of adversarial examples
        :rtype: n x m tensor
        """
        for epoch in range(self.epochs):
            surface.backward(x, y, cov=self.change_of_variables)
            if self.change_of_variables:
                self.tanh_map(x, into=True)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if self.change_of_variables:
                self.tanh_map(x, into=False)
            x.clamp_(*self.x_range)
        return x

    def initialize(self, x, surface):
        """
        This method performs any preprocessing and initialization steps prior
        to crafting adversarial examples. Specifically, some attacks (1)
        initialize x via a random perturbation (e.g., PGD), (2) require the
        original x in loss functions (e.g., CW), or (3) embed perturbations
        magnitudes directly in gradients (e.g., l2-based attacks). This method
        performs such preprocessing subroutines, as well as attaches x to the
        associated optimizer.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: n x m tensor
        :param surface: surface containing the parameters to optimize over
        :type surface: Surface object
        :return: None
        :rtype: NoneType
        """

        # apply random restart
        x.add_(
            (
                torch.distributions.uniform.Uniform(
                    -self.random_alpha, self.random_alpha
                ).sample(x.size())
            )
            if self.random_alpha
            else 0
        ).clamp_(*self.x_range)

        # override alpha for l2 attacks, and attach x to loss & optimizer
        self.alpha = (
            1.0
            if surface.norm == 2 and self.optimizer == torch.optim.SGD
            else self.alpha
        )
        surface.loss.attach(x) if surface.loss.x_req else None
        # if CoV is true, transform x to w
        # x = (
        #     x.mul_(2).sub_(1).mul_(self.tanh_smoother).arctanh_()
        #     if self.change_of_variables is True
        #     else x
        # )
        self.opt = self.optimizer([x], lr=self.alpha)
        self.init_req = False
        return None

    def tanh_map(self, x, into=True):
        """
        This method  maps x into the tanh-space, as shown in
        https://arxiv.org/pdf/1608.04644.pdf. Specifically, the transformation
        is defined as follows:

                            Δ = 1/2 * (Tanh(w) + 1) - x

        where w is the variable being optimized over, x is the original input,
        and  Δ is the resultant perturbation to be added to x to produce the
        adversarial example. To perform such a transformation, we do this in
        two steps: (1) x is first mapped into the ArcTanh-space, and (2) when
        optimizing for w, w + x is mapped into the Tanh-space (i.e., Δ). Since
        step (2) is done during the optimization step, this function defines an
        "into" mode that maps into the ArcTanh-space when true, and maps back
        out (via Tanh) when false. Importantly, this method operates in-place,
        as x often needs to be attached to the optimizer.

        :param x: the batch of inputs to map into tanh-space
        :type x: n x m tensor
        :param into: whether to map into (or out of) the arctanh-space
        :type into: boolean
        :return: x mapped into (or back out of) tanh-space
        :rtype: n x m tensor
        """
        return (
            x.mul_(2).sub_(1).mul_(self.tanh_smoother).arctanh_()
            if into
            else x.tanh_().add_(1).div_(2)
        )


if __name__ == "__main__":
    """
    Example usage from [paper_url].
    """
    raise SystemExit(0)
