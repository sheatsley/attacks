"""
This module defines functions to instantiate PyTorch-based deep learning models.
Author: Ryan Sheatsley & Blaine Hoak
Mon Jul 6 2020
"""

import attacks  # PyTorch-based framework for attacking deep learning models
import itertools  # Functions creating iterators for efficient looping
import loss as libloss  # PyTorch-based custom loss functions
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
from utilities import print  # Use timestamped print

# TODO
# consider better debugging when field in architectures is not specified (eg adv_train)
# remove attack object instantiation when Attacks module is updated
# reconsider inheritence structure (all args must be passed to each class atm)


class LinearClassifier(torch.nn.Module):
    """
    This class defines a simple PyTorch-based linear model class that contains
    the fundamental components of any deep neural network. While this is class
    is designed to be inherited to provide simple sub-classes for multi-layer
    linear or convolutional models, instantiating this class with default
    parameters instantiates a single-layer linear model with categorical
    cross-entropy loss. It defines the following methods:

    :func:`__init__`: initial object setup
    :func:`__call__`: returns model logits
    :func:`accuracy`: computes the accuracy of the model on data
    :func:`cpu`: moves all tensors to the cpu
    :func:`build`: assembles the PyTorch model
    :func:`fit`: trains the model
    :func:`freeze`: disable autograd tracking of model parameters
    :func:`load`: loads saved model parameters
    :func:`predict`: returns predicted labels
    """

    def __init__(
        self,
        adv_train=None,
        optimizer=torch.optim.Adam,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=1e-4,
        iters=10,
        threads=None,
        info=0.25,
    ):
        """
        This method describes the initial setup for a simple linear classifier.
        Importantly, models are not usable until they are trained (via fit()),
        this "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to scikit-learn
        model classes) on initialization.

        :param adv_train: attack used for adversarial training
        :type adv_train: Attack object
        :param optimizer: optimizer from torch.optim package
        :type optimizer: callable
        :param loss: loss function from torch.nn.modules.loss
        :type loss: callable
        :param batch_size: size of minibatches
        :type batch_size: integer
        :param learning_rate: learning rate schedule
        :type learning_rate: float
        :param iters: number of training iterations (ie epochs)
        :type iters: integer
        :param threads: sets the threads used for training
        :type threads: integer
        :param info: print the loss every info percent
        :type info: float between 0 and 1
        :return: Single-layer classifier
        :rtype: LinearClassifier object
        """
        super().__init__()
        self.adv_train = adv_train
        self.optimizer = optimizer
        self.loss = loss()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = iters
        self.threads = (
            min(threads, torch.get_num_threads())
            if threads
            else torch.get_num_threads()
        )
        self.info = max(1, int(iters * info))
        return None

    def __call__(self, x, grad=True):
        """
        This method allows us to have MLPClassifier objects behave as
        functions directly. Specifically, we redefine this method to pass
        Tensor inputs directly into Pytorch Sequential containers, which allows
        class objects to behave more like augmented Pytorch models directly
        instead of custom classes that embedd Pytorch models inside.

        :param x: samples
        :type x: n x m tensor of samples
        :param grad: whether or not to keep track of operations on x
        :type grad: boolean
        :return: model logits
        :rtype: n x c tensor where c is the number of classes
        """
        with torch.set_grad_enabled(grad):
            return self.model(x)

    def accuracy(self, x, y):
        """
        This method simply returns the fraction of samples classified correctly
        (as defined by y) over the total number of samples.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: model accuracy
        :rtype: float
        """
        return torch.eq(torch.argmax(self(x, grad=False), dim=1), y).sum() / y.numel()

    def build(self, x, y):
        """
        This method instantiates a PyTorch sequential container. This
        abstraction allows us to dynamically build models based on the
        passed-in dataset, as opposed to hardcoding model architectures via
        defining a "forward" method.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: linear model
        :rtype: Sequential container
        """
        return torch.nn.Sequential(torch.nn.Linear(x.size(1), torch.unique(y).size(0)))

    def cpu(self):
        """
        This method moves both the model and the optimizer state to the cpu.
        At this time, `to`, `cpu`, and `cuda` methods are not supported for
        optimizers, so we must manually set tensor devices
        (https://github.com/pytorch/pytorch/issues/41839).

        :return the model itself
        :rtype: LinearClassifier object
        """
        self.model.cpu()
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()
        return self

    def fit(self, x, y):
        """
        This method prepends and appends linear layers with dimensions
        inferred from the dataset. Moreover, it trains the model using the
        paramterized optimizer.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :param dataset: samples with associated labels
        :type dataset: (n x m, n) tuple of tensors as a TensorDataset
        :return: a trained model
        :rtype: LinearClassifier object
        """

        # instantiate learning model and save dimensionality
        self.model = self.build(x, y)
        self.features = x.size(1)
        print(f"Defined model:\n{self.model}")

        # configure optimizer and load data into data loader
        print(f"Preparing data loader of shape: {x.size()} x {y.size()}")
        self.opt = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda" if x.is_cuda else "cpu"),
        )

        # attatch loss and model parameters to attacks for adversarial training
        if self.adv_train is not None:
            self.adv_train = attacks.Attack(
                **{
                    **self.adv_train,
                    **{
                        "model": self,
                        "loss": libloss.Loss(
                            type(self.loss), max_obj=True, x_req=False, reduction="none"
                        ),
                    },
                }
            )
            print(
                "Performing",
                self.adv_train.name,
                "adversarial training!",
                f"({self.adv_train.traveler.epochs} epochs,",
                f"alpha={self.adv_train.traveler.alpha})",
            )

        # main training loop
        print(f"Unfreezing parameters and limiting thread count to {self.threads}...")
        self.model.train()
        self.freeze(False)
        max_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        for epoch in range(self.epochs):
            eloss = 0
            for sample_batch, label_batch in dataset:

                # disable grad tracking for model parameters
                if self.adv_train is not None:
                    self.freeze(True)
                    sample_batch = self.adv_train.craft(sample_batch, label_batch)
                    self.adv_train.traveler.init_req = True
                    self.freeze(False)
                loss = self.loss(
                    self.model(sample_batch),
                    label_batch,
                )
                loss.backward()
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                eloss += loss.item()

            # show model loss every info-percent
            if not epoch % self.info:
                print(f"Loss at epoch {epoch}: {eloss}")
        accuracy = self.accuracy(x, y).item()
        print(f"Final loss: {eloss:.2f}, & accuracy: {accuracy:.2f}")

        # print statistics about adversarial training
        if self.adv_train is not None:
            train_adv = self.adv_train.craft(x, y)
            adv_loss = self.loss(self.model(train_adv), y)
            adv_acc = self.accuracy(train_adv, y).item()
            print(f"Adversarial loss: {adv_loss:.2f} & accuracy: {adv_acc:.2f}")
        print(f"Freezing parameters and restoring thread count to {max_threads}...")
        torch.set_num_threads(max_threads)
        self.freeze(True)
        self.model.eval()
        return self

    def freeze(self, freeze=True):
        """
        This method iterates over all model parameters and disables tracking
        operation history within the autograd system. This is particularly
        useful in cases with adversarial machine learning: operations on inputs
        are tracked (as opposed to model parameters when training) and
        gradients with respect to these inputs are computed. Since they are
        multiple backward()'s calls in these scenarios, we can achieve
        (sometimes significant) performance gains by explicitly disabling
        autograd history for model parameters.

        :param freeze: enable or disable autograd history for model parameters
        :type freeze: boolean
        :return: the model itself
        :rtype: LinearClassifier object
        """
        for param in self.model.parameters():
            param.requires_grad = not freeze
        return self

    def load(self, path, x, y):
        """
        This method loads a set of model parameters, saved via torch.save
        method. Since this class defines PyTorch models via the build method,
        we also require the training data and labels to properly infer input
        feautres and output labels.

        :param path: path to the saved model parameters
        :type path: string
        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: the model itself
        :rtype: LinearClassifier object
        """
        self.model = self.build(x, y)
        self.features = x.size(1)
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded model:\n{self.model}")
        return self

    def predict(self, x):
        """
        This method serves as a wrapper for __call__ to behave like predict
        methods in scikit-learn model classes by returning an class label
        (instead of model logits).

        :param x: samples
        :type x: n x m tensor of samples
        :return: predicted labels
        :rtype: n-length tensor of integers
        """
        return torch.argmax(self(x, grad=False), dim=1) if len(x) else torch.tensor([])


class MLPClassifier(LinearClassifier):
    """
    This class inherits from LinearClassifier and instantiates a PyTorch-based
    multi-layer (if the number of hidden layers is non-zero, else a
    single-layer is returned) perceptron classifier. Specifically, it inherits
    the following methods as-is from LinearClassifier:

    :func:`__init__`: initial class setup
    :func:`__call__`: returns model logits
    :func:`cpu`: moves all tensors to the cpu
    :func:`fit`: trains the model
    :func:`freeze`: disable autograd tracking of model parameters
    :func:`load`: loads saved model parameters
    :func:`predict`: returns predicted labels

    It redefines the following methods:

    :func:`build`: assembles the PyTorch MLP model
    """

    def __init__(
        self,
        hidden_layers=(15,),
        activation=torch.nn.ReLU,
        adv_train=None,
        optimizer=torch.optim.Adam,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=1e-4,
        iters=10,
        threads=None,
        info=0.25,
    ):
        """
        This function describes the initial setup for a multi-layer perceptron
        classifier. Importantly, models are not usable until they are trained
        (via fit()), this "lazy" model creation schema allows us to abstract
        out parameterization of number of attributes and labels (thematically
        similar to scikit-learn model classes).

        :param hidden_layers: the number of neurons at each layer
        :type hidden_layers: i-length tuple of integers
        :param activation: activation functions from torch.nn.functional
        :type activation: callable
        :param adv_train: attack used for adversarial training
        :type adv_train: Attack object
        :param optimizer: supports optimizers in torch.optim package
        :type optimizer: callable from torch.optim
        :param loss: supports loss functions in torch.nn
        :type loss: callable from torch.nn.modules.loss
        :param batch_size: size of minibatches
        :type batch_size: integer
        :param learning_rate: learning rate schedule
        :type learning_rate: float
        :param iters: number of training iterations (ie epochs)
        :type iters: integer
        :param threads: sets the threads used for training
        :type threads: integer
        :param info: print the loss at % itervals
        :type info: float between 0 and 1
        :return: mutli-layer linear classifier
        :rtype: MLPClassifier object
        """
        super().__init__(
            adv_train, optimizer, loss, batch_size, learning_rate, iters, threads, info
        )
        self.hidden_layers = hidden_layers
        self.activation = activation
        return None

    def build(self, x, y):
        """
        This method overrides the implementation of build() from the
        LinearClassifier class. Specifically, it adds support for hidden
        layers and activation functions when instantiating a PyTorch sequential
        container. The provided abstraction allows us to dynamically build
        models based on the passed-in dataset, as opposed to hardcoding model
        architectures via defining a "forward" method.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: multi-layer perceptron model
        :rtype: Sequential container
        """

        # compute the number of classes (needed now if no hidden layers)
        labels = torch.unique(y).size(0)

        # instantiate initial & hidden linear layers
        neurons = (x.size(1),) + self.hidden_layers
        linear = (
            [
                torch.nn.Linear(neurons[i], neurons[i + 1])
                for i in range(len(neurons) - 1)
            ]
            if len(self.hidden_layers)
            else [torch.nn.Linear(x.size(1), labels)]
        )

        # interleave linear layers with activation function
        layers = itertools.product(
            linear,
            [self.activation()],
        )

        # add output layer and instantiate sequential container
        return torch.nn.Sequential(
            *itertools.chain(
                itertools.chain.from_iterable(layers),
                [torch.nn.Linear(self.hidden_layers[-1], labels)]
                if len(self.hidden_layers)
                else [],
            )
        )


class CNNClassifier(LinearClassifier):
    """
    This class inherits from LinearClassifier and instantiates a PyTorch-based
    convolutional neural network classifier. Specifically, it inherits the
    following methods as-is from LinearClassifier:

    :func:`__init__`: initial class setup
    :func:`__call__`: returns model logits
    :func:`cpu`: moves all tensors to the cpu
    :func:`fit`: trains the model
    :func:`freeze`: disable autograd tracking of model parameters
    :func:`load`: loads saved model parameters
    :func:`predict`: returns predicted labels

    It redefines the following methods:

    :func:`build`: assembles the PyTorch CNN model
    """

    def __init__(
        self,
        conv_layers,
        shape=None,
        stride=1,
        kernel_size=3,
        linear_layers=(15,),
        drop_prob=0.5,
        activation=torch.nn.ReLU,
        adv_train=None,
        optimizer=torch.optim.Adam,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=1e-4,
        iters=10,
        threads=None,
        info=0.25,
    ):
        """
        This function describes the initial setup for a  convolutional neural
        network classifier. Importantly, models are not usable until they are
        trained (via fit()), this "lazy" model creation schema allows us to
        abstract out parameterization of number of attributes and labels
        (thematically similar to scikit-learn model classes).

        :param conv_layers: the number of filters at the ith layer
        :type conv_layers: i-length tuple
        :param shape: the expected shape of the input image
        :type shape: tuple of form: (channels, width, height)
        :param stride: stride length for the convolutional layers
        :type stride: integer
        :param kernel_size: kernel size for all convolutional layers
        :type kernel_size: integer
        :param linear_layers: the number of neurons at each linear layer
        :type linear_layers: i-length tuple of integers
        :param drop_prob: probability of an element to be zeroed (omitted if 0)
        :type drop_prob: float
        :param activation: activation functions from torch.nn.functional
        :type activation: callable
        :param adv_train: attack used for adversarial training
        :type adv_train: Attack object
        :param optimizer: supports optimizers in torch.optim package
        :type optimizer: callable from torch.optim
        :param loss: supports loss functions in torch.nn
        :type loss: callable from torch.nn.modules.loss
        :param batch_size: size of minibatches
        :type batch_size: integer
        :param learning_rate: learning rate schedule
        :type learning_rate: float
        :param iters: number of training iterations (ie epochs)
        :type iters: integer
        :param threads: sets the threads used for training
        :type threads: integer
        :param info: print the loss at % itervals
        :type info: float between 0 and 1
        :return: convolutional neural network classifier
        :rtype: CNNClassifier object
        """
        super().__init__(
            adv_train, optimizer, loss, batch_size, learning_rate, iters, threads, info
        )
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.linear_layers = linear_layers
        self.activation = activation
        self.shape = shape

        # set popular maxpool params & enable benchmarks
        self.mp_ksize = 2
        self.mp_stride = 2
        torch.backends.cudnn.benchmark = True
        return None

    def build(self, x, y):
        """
        This method overrides the implementation of build() from the
        LinearClassifier class. Specifically, it adds support for
        convolutional, dropout, and hidden layers, as well as activation
        functions when instantiating a PyTorch sequential container. The
        provided abstraction allows us to dynamically build models based on the
        passed-in dataset, as opposed to hardcoding model architectures via
        defining a "forward" method.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: convolutional model
        :rtype: Sequential container
        """

        # compute the number of classes and output of last maxpool
        labels = torch.unique(y).size(0)
        if not self.shape:
            self.shape = x.size()[1:]
            x = x.flatten()
        last_maxout = (
            self.shape[1] // self.mp_ksize ** len(self.conv_layers)
        ) ** 2 * self.conv_layers[-1]

        # instantiate convolutional layers
        conv_layers = (self.shape[0],) + self.conv_layers
        conv_layers = [
            torch.nn.Conv2d(
                conv_layers[i],
                conv_layers[i + 1],
                self.kernel_size,
                self.stride,
                "same",
            )
            for i in range(len(conv_layers) - 1)
        ]

        # instantiate linear layers
        neurons = (last_maxout,) + self.linear_layers
        linear_layers = (
            [
                torch.nn.Linear(neurons[i], neurons[i + 1])
                for i in range(len(neurons) - 1)
            ]
            if len(self.linear_layers)
            else [torch.nn.Linear(last_maxout, labels)]
        )

        # interleave initial and convolution layers with activation and maxpool
        cnn_layers = itertools.product(
            conv_layers,
            [self.activation()],
            [
                torch.nn.MaxPool2d(kernel_size=self.mp_ksize, stride=self.mp_stride),
            ],
        )

        # interleave output of maxpool and linear layers with activation
        mlp_layers = itertools.product(
            linear_layers,
            [self.activation()],
        )

        # concatenate cnn & mlp layers, add dropout, and return the container
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, self.shape),
            *itertools.chain.from_iterable(cnn_layers),
            torch.nn.Flatten(),
            torch.nn.Dropout(self.drop_prob, inplace=True),
            *itertools.chain(
                itertools.chain.from_iterable(mlp_layers),
                [torch.nn.Linear(self.linear_layers[-1], labels)]
                if len(self.linear_layers)
                else [],
            ),
        )


if __name__ == "__main__":
    """
    Example usage with training MNIST MLP and CNN models.
    """
    import architectures  # optimal PyTorch-based model architectures and hyperparameters
    import sklearn.metrics  # Score functions, performance metrics, and pairwise metrics and distance computations
    import utilities  # Various utility functions

    # load dataset
    use_cuda, device = (True, "cuda:0") if torch.cuda.is_available() else (False, "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor) if use_cuda else None
    tx, ty, vx, vy = utilities.load("mnist", device=device)

    # train linear model
    print("Training MNIST Linear Classifier...")
    linear_model = LinearClassifier(
        iters=6,
        learning_rate=1e-3,
    ).fit(tx, ty)
    print(
        "Linear Training Report:\n",
        sklearn.metrics.classification_report(ty.cpu(), linear_model.predict(tx).cpu()),
        "\nLinear Testing Report:\n",
        sklearn.metrics.classification_report(vy.cpu(), linear_model.predict(vx).cpu()),
    )

    # train mlp model
    print("Training MNIST MLP Classifier...")
    mlp_model = MLPClassifier(**architectures._mnist(cnn=False)).fit(tx, ty)
    print(
        "MLP Training Report:\n",
        sklearn.metrics.classification_report(ty.cpu(), mlp_model.predict(tx).cpu()),
        "\nMLP Testing Report:\n",
        sklearn.metrics.classification_report(vy.cpu(), mlp_model.predict(vx).cpu()),
    )

    # train cnn model
    print("Training MNIST CNN Classifier...")
    cnn_model = CNNClassifier(**architectures.mnist).fit(tx, ty)
    print(
        "CNN Training Report:\n",
        sklearn.metrics.classification_report(ty.cpu(), cnn_model.predict(tx).cpu()),
        "\nCNN Testing Report:\n",
        sklearn.metrics.classification_report(vy.cpu(), cnn_model.predict(vx).cpu()),
    )
    raise SystemExit(0)
