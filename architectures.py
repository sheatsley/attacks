"""
This module defines the state-of-the-art architectures and hyperparameters for
a variety of datasets from https://github.com/sheatsley/datasets. It strives to
be a useful form of parameter bookkeeping (in the form of dictionaries) to be
passed directly as arguments to model initializations (from the learn module).
various datasets. Specifically, it encodes model architectures,
hyperparameters, activation functions, and other miscellaneous details.
Author: Ryan Sheatsley and Blaine Hoak
Thu Apr 14 2022
"""
import attacks  # PyTorch-based framework for attacking deep learning models
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add unit test


def __getattr__(dataset):
    """
    PEP 562 (https://www.python.org/dev/peps/pep-0562) gives us a clever way to
    call module functions in an attribute-style (without instantiating an
    object), which is thematic with the purpose of this module.

    :param dataset: the dataset to retrieve arguments from
    :type dataset: string; one of the method defined below
    :return: the dataset architecture and hyperparameters
    :rtype: dictionary of arguments and values
    """
    return globals()[f"_{dataset.lower()}"]()


def _drebin(cnn=False):
    """
    The Drebin Dataset is an Android malware dataset with malicious and benign
    samples. It has six features that describe the frequencies of behaviors
    present in the APK. The current state-of-the-art accuracy is over 99%
    (https://arxiv.org/pdf/1805.11843.pdf). The parameters below are for a
    multi-layer perceptron.
    """
    return {}


def _fmnist(cnn=True):
    """
    Fashion-MNIST is a dataset for predicting Zalando's article images. It has
    ten labels that describe particular articles of clothing, encoded as "0"
    through "9" (i.e., t-shirt/top, trouser, pullover, dress, coat, sandal,
    shirt, sneaker, bag, ankle boot). It is designed as a drop-in replacement
    for the original MNIST dataset for benchmarking machine learning
    algorithms. The state-of-the-art accuracy is over 99%
    (https://arxiv.org/pdf/2001.00526.pdf). As the dataset is naturally of
    images, the default parameters are for a convolutional neural network,
    while, when the "cnn" argument is false, competitive parameters for
    multi-layer perceptrons are returned instead.

    :param cnn: return parameters for CNNs (else MLPs)
    :type cnn: boolean
    :return: SOTA parameters for vanilla CNNs (or MLPs)
    :rtype: dict; keys are arguments, while values are hyperparameter values
    """
    return (
        {
            "activation": torch.nn.ReLU,
            "adv_train": {
                "epochs": 30,
                "optimizer": torch.optim.SGD,
                "alpha": 0.01,
                "random_alpha": 0.1,
                "change_of_variables": False,
                "saliency_map": "identity",
                "norm": float("inf"),
                "jacobian": "model",
            },
            "batch_size": 64,
            "conv_layers": (16, 32),
            "drop_prob": 0.4,
            "iters": 20,
            "kernel_size": 3,
            "learning_rate": 1e-3,
            "linear_layers": (512,),
            "loss": torch.nn.CrossEntropyLoss,
            "optimizer": torch.optim.Adam,
            "shape": (1, 28, 28),
            "stride": 1,
            "threads": 8,
        }
        if cnn
        else {
            "activation": torch.nn.ReLU,
            "batch_size": 64,
            "optimizer": torch.optim.Adam,
            "hidden_layers": (
                512,
                256,
                100,
            ),
            "iters": 8,
            "learning_rate": 1e-2,
            "loss": torch.nn.CrossEntropyLoss,
            "threads": 6,
        }
    )


def _nslkdd(cnn=None):
    """
    The NSL-KDD contains extracted feature vectors from PCAPs that contain
    various information about traffic flows. It has five labels that describe
    benign traffic, denial-of-service attacks, network probes, user-to-root
    attacks, and remote-to-local attacks. The current state-of-the-art accuracy
    is ~82% (https://www.ee.ryerson.ca/~bagheri/papers/cisda.pdf). The
    parmaeters below are for a multi-layer perceptron.
    """
    return {
        "activation": torch.nn.ReLU,
        "adv_train": {
            "epochs": 10,
            "optimizer": torch.optim.SGD,
            "alpha": 0.01,
            "random_alpha": 0.01,
            "change_of_variables": False,
            "saliency_map": "identity",
            "norm": float("inf"),
            "jacobian": "model",
        },
        "batch_size": 128,
        "optimizer": torch.optim.Adam,
        "hidden_layers": (60, 32),
        "iters": 4,
        "learning_rate": 1e-2,
        "loss": torch.nn.CrossEntropyLoss,
        "threads": 8,
    }


def _mnist(cnn=True):
    """
    MNIST is a dataset for predicting handwritten digits. It has ten labels
    that describe a particular digit, "0" through "9". The state-of-the-art
    accuracy is over 99% (https://arxiv.org/pdf/1710.09829.pdf). As the dataset
    is naturally of images, the default parameters are for a convolutional
    neural network, while, when the "cnn" argument is false, competitive
    parameters for multi-layer perceptrons are returned instead. Finally, the
    adversarial training regime is based on the state-of-the-art adversarial
    robustness measures with standard attacks  which 89.3% against a PGD-based
    adversary with 100 steps and 20 random-restarts
    (https://arxiv.org/pdf/1706.06083.pdf).

    :param cnn: return parameters for CNNs (else MLPs)
    :type cnn: boolean
    :return: SOTA parameters for vanilla CNNs (or MLPs)
    :rtype: dict; keys are arguments, while values are hyperparameter values
    """
    return (
        {
            "activation": torch.nn.ReLU,
            "adv_train": {
                "epochs": 30,
                "optimizer": torch.optim.SGD,
                "alpha": 0.01,
                "random_alpha": 0.1,
                "change_of_variables": False,
                "saliency_map": "identity",
                "norm": float("inf"),
                "jacobian": "model",
            },
            "batch_size": 64,
            "conv_layers": (16, 32),
            "drop_prob": 0.4,
            "iters": 20,
            "kernel_size": 3,
            "learning_rate": 1e-3,
            "linear_layers": (128,),
            "loss": torch.nn.CrossEntropyLoss,
            "optimizer": torch.optim.Adam,
            "shape": (1, 28, 28),
            "stride": 1,
            "threads": 8,
        }
        if cnn
        else {
            "activation": torch.nn.ReLU,
            "batch_size": 64,
            "optimizer": torch.optim.Adam,
            "hidden_layers": (512,),
            "iters": 20,
            "learning_rate": 1e-2,
            "loss": torch.nn.CrossEntropyLoss,
            "threads": 6,
        }
    )


def _phishing(cnn=False):
    """
    This anti-phishing websites dataset is for predicting whether or not a
    website is malicious. The feature span areas from the website DOM and URL.
    The state-of-the-art accuracy is 96%
    (https://www.sciencedirect.com/science/article/pii/S0020025519300763). The
    parameters below are for a multi-layer perceptron.
    """
    return {
        "activation": torch.nn.ReLU,
        "adv_train": {
            "epochs": 10,
            "optimizer": torch.optim.SGD,
            "alpha": 0.01,
            "random_alpha": 0.05,
            "change_of_variables": False,
            "saliency_map": "identity",
            "norm": float("inf"),
            "jacobian": "model",
        },
        "batch_size": 32,
        "optimizer": torch.optim.Adam,
        "hidden_layers": (15,),
        "iters": 40,
        "learning_rate": 1e-2,
        "loss": torch.nn.CrossEntropyLoss,
        "threads": 8,
    }


def _unswnb15(cnn=False):
    """
    The UNSW-NB15 dataset is for predicting network intrusions from a blend of
    real benign traffic with synthetically generated attacks. It contains nine
    different attacks of varying types. The current state-of-the-art accuracy
    is 81%
    (https://www.sciencedirect.com/science/article/pii/S0957417419300843). The
    parameters below are for a multi-layer perceptron.
    """
    return {
        "activation": torch.nn.ReLU,
        "adv_train": {
            "epochs": 5,
            "optimizer": torch.optim.SGD,
            "alpha": 0.01,
            "random_alpha": 0.01,
            "change_of_variables": False,
            "saliency_map": "identity",
            "norm": float("inf"),
            "jacobian": "model",
        },
        "batch_size": 128,
        "optimizer": torch.optim.Adam,
        "hidden_layers": (15,),
        "iters": 40,
        "learning_rate": 1e-2,
        "loss": torch.nn.CrossEntropyLoss,
        "threads": 8,
    }


if __name__ == "__main__":
    """
    Prints parameters for all datasets (useful for debugging).
    """
    raise SystemExit(0)
