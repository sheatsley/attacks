"""
This module defines custom Pytorch-based optimizers.
Authors: Blaine Hoak & Ryan Sheatsley
Wed Jun 29 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# implement MBS
# implmenet BWSGD


class SGD(torch.optim.SGD):
    """
    This class is identical to the SGD class in PyTorch, with the exception
    that two additional attributes are added: req_loss (set to False) and
    req_acc (set to False). Unlike SGD, some optimizers (e.g., MBS & BWSGD)
    either require the current model loss or accuracy to perform their update
    step, and thus, we instantiate these parameters for a homogenous interface.

    :func:`__init__` instantiates an SGD object
    """

    def __init__(self, **kwargs):
        """
        This method instantiates an SGD object. It accepts keyword arguments
        for the PyTorch parent class described in:
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.

        :param kwargs: keyword arguments for torch.optim.SGD
        :type kwargs: dict
        :return Stochastic Gradient Descent optimizer
        :rtype: SGD object
        """
        super().__init__(**kwargs)
        self.req_loss = False
        self.req_acc = False
        return None


class Adam(torch.optim.Adam):
    """
    This class is identical to the Adam class in PyTorch, with the exception
    that two additional attributes are added: req_loss (set to False) and
    req_acc (set to False). Unlike Adam, some optimizers (e.g., MBS & BWSGD)
    either require the current model loss or accuracy to perform their update
    step, and thus, we instantiate these parameters for a homogenous interface.

    :func:`__init__` instantiates an Adam object
    """

    def __init__(self, **kwargs):
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
        self.req_loss = False
        self.req_acc = False
        return None
