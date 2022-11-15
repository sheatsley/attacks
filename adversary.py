"""
This modules defines the adversaries proposed in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Fri Oct 21 2022
"""
import attacks  # Attacks on machine learning in PyTorch
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
import sklearn.preprocessing  # Preprocessing and Normalization


class Adversary:
    """
    The Adversary class supports generalized threat modeling. Specifically, as
    shown in https://arxiv.org/pdf/1608.04644.pdf and
    https://arxiv.org/pdf/1907.02044.pdf, effective adversaries often enact
    some decision after adversarial examples have been crafted. For the
    provided examples, such adversaries embed hyperparamter optimization, and
    perform multiple restarts (useful only if the attack is non-deterministic)
    and simply return the most effective adversarial examples. This class
    facilitiates such functions.

    :func:`__init__`: instantiates Adversary objects
    :func:`__repr__`: returns the threat model
    :func:`attack`: returns a set of adversarial examples
    :func:`minmaxscale`: normalizes inputs to be within [0, 1]
    """

    def __init__(
        self,
        hparam,
        hparam_criterion,
        hparam_steps,
        max_loss_min_norm,
        num_restarts,
        **atk_args,
    ):
        """
        This method instantiates an Adversary object with a hyperparameter to
        be optimized (via binary search), the criterion to control the
        hyperparameter search, the number of hyperparameter optimization steps,
        whether the adversary aims to maximize loss or minimize norm (while
        still misclassifying inputs), and the number of restarts.

        :param hparam: the hyperparameter to be optimized
        :type hparam: str
        :param hparam_criterion: criterion for hyperparameter optimization
        :type hparam_criterion: callable
        :param hparam_steps: the number of hyperparameter optimization steps
        :type hparam_steps: int
        :param num_restarts: number of restarts to consider
        :type num_restarts: int
        :param restart_criterion: criterion for keeping inputs across restarts
        :type restart_criterion: callable
        :param atk_args: attack parameters
        :type atk_args: dict
        :return: an adversary
        :rtype: Adversary object
        """
        self.hparam = hparam
        self.hparam_criterion = hparam_criterion
        self.hparam_steps = hparam_steps
        self.max_loss_min_norm = self.max_loss_min_norm
        self.num_restarts = num_restarts
        self.attack = attacks.Attack(**atk_args)
        return None

    def minmaxscale(self, x, transform=True):
        """
        This method serves as a wrapper for sklearn.preprocessing.MinMaxScaler.
        Specifically, it maps inputs to [0, 1] (as some techniques assume this
        range, e.g., change of variables). Importantly, since these scalers
        always return numpy arrays, this method additionally casts these inputs
        back as PyTorch tensors with the original data type.

        :param x: the batch of inputs to scale
        :type x: PyTorch FloatTensor object (n, m)
        :param transform: performs the transformation if true; the inverse if false
        :return: a batch of scaled inputs
        :rtype: PyTorch FloatTensor (n, m)
        """
        if transform:
            self.scaler = sklearn.preprocessing.MinMaxScaler()
            self.scaler.dtype = x.dtype
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.inverse_transform(x)
        return torch.from_numpy(x).to(self.scaler.dtype)


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
