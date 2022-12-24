"""
This module defines the attacks proposed in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Mon Apr 18 2022
"""
import aml.loss as loss  # Popular PyTorch-based loss functions
import aml.optimizer as optimizer  # PyTorch-based custom optimizers for crafting adversarial examples
import aml.saliency as saliency  # Gradient manipulation heuristics to achieve adversarial goals
import aml.surface as surface  # PyTorch-based models for crafting adversarial examples
import aml.traveler as traveler  # PyTorch-based perturbation schemes for crafting adversarial examples
import itertools  # Functions creating iterators for efficietn looping
import sklearn.preprocessing  # Preprocessing and Normalization
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# implement unit test
# add early termination support
# implement DDN (implement alternating SGD optimizer)
# cleanup hparam updates
# consider merging min_norm and carlini interfaces
# epsilon and clipping  (min_p, max_p) do not work for l0, maybe l2 as well
# ^ need to recompute clips at each iteration if we're going to depend on distance "left"
# add perturbations visualizations in examples directory
# have option to silence all prints (good for unittest)
# address statistics tracking when batch size isn't -1
# l2 rr should be normlized by l2-norm and l0 norm should pick max l0 random features
# write l0, l2, and linf specialized clamps (eg, project l2 vector to budget instead of 1)


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
    :func:`carlini`: hyperparameter criterion used in CW attack
    :func:`max_loss`: restart criterion used in PGD attack
    :func:`min_norm`: restart criterion used in FAB attack
    :func:`minmax_scale`: normalizes inputs to be within [0, 1]
    """

    def __init__(
        self,
        hparam,
        hparam_bounds,
        hparam_criterion,
        hparam_steps,
        num_restarts,
        restart_criterion,
        **atk_args,
    ):
        """
        This method instantiates an Adversary object with a hyperparameter to
        be optimized (via binary search), the lower and upper bounds to
        consider for binary search, the criterion to control the hyperparameter
        search (by convention, "success" is encoded as True), the number of
        hyperparameter optimization steps, the number of restarts, and the
        criterion to determine if an input is kept across restarts.

        :param hparam: the hyperparameter to be optimized
        :type hparam: str
        :param hparam_bounds: lower and upper bounds for binary search
        :type hparam_bounds: tuple
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
        attack = Attack(**atk_args)
        self.hparam = hparam
        self.hparam_bounds = hparam_bounds
        self.hparam_criterion = hparam_criterion
        self.hparam_steps = min(hparam_steps, 1)
        self.num_restarts = min(num_restarts, 1)
        self.restart_criterion = restart_criterion
        self.attack = attack
        self.loss = atk_args["loss"]
        self.norm = atk_args["norm"]
        self.model = atk_args["model"]
        self.params = {
            "hparam": hparam,
            "hparam bounds": hparam_bounds,
            "hparam criterion": hparam_criterion.__name__,
            "hparam steps": hparam_steps,
            "num restarts": num_restarts,
            "restart_criterion": restart_criterion.__name__,
            "attack": attack.name,
        }
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of adversary
        parameters and the attack name.

        :return: adversarial parameters and the attached attack name
        :rtype: str
        """
        return f"Adversary({self.params})"

    def attack(self, x, y):
        """
        This method represents the core of the Adversary class. Specifically,
        this class produces the most effective adversarial examples,
        parameterized by restarts and hyperparameter optimization. Notably, as
        criterion callables can require arbitrary arguments, it is assumed all
        such callables accept keyword arguments.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: a batch of adversarial examples
        :rtype: torch Tensor object (n, m)
        """

        # rescale inputs and  iterate over restarts & hyperparameters
        print("Rescaling inputs...")
        x = self.minmaxscale(x)
        best_p = torch.zeros_like_(x)
        for iter_step in range(self.num_restarts):
            print(
                f"On iteration {iter+1} of {self.num_restarts}...",
                f"({iter_step/self.num_restarts:.1%})",
            )

            # instantiate bookkeeping and apply hyperparameter criterion
            prev_p = torch.full_like(x, torch.inf)
            p = torch.zeros_like(x)
            hparam_bounds = torch.tensor(self.hparam_bounds).repeat(y.numel())
            for hparam_step in range(self.hparam_steps):
                curr_p = self.attack(x + p, y)
                if self.hparam:
                    print(
                        "On hyperparameter iteration {hparam_step+1}} of",
                        "{self.hparam_steps}...",
                        f"({hparam_step/self.hparam_steps:.1%})",
                    )
                    hparam_value = self.attack.hparams[self.hparam]
                    hparam_updates = self.hparam_criterion(x, curr_p, prev_p, y)
                    hparam_bounds[hparam_updates][1] = torch.minimum(
                        hparam_bounds[hparam_updates][1], hparam_value[hparam_updates]
                    )
                    hparam_bounds[~hparam_updates][0] = torch.maximum(
                        hparam_bounds[~hparam_updates][0], hparam_value[~hparam_updates]
                    )
                    self.attack.hparams[self.hparam] = hparam_bounds.diff().div(2)
                    print(
                        f"{hparam_updates.sum().div(hparam_updates.numel()):.2%}",
                        "inputs were determined succesful.",
                        f"Hyperparameter {self.hparam} updated.",
                    )

                # store the best adversarial examples seen thus far
                best_updates = self.restart_criterion(x, curr_p, best_p, y)
                best_p[best_updates] = curr_p[best_updates]
                print(
                    f"Found {best_updates.sum()} better adversarial examples!",
                    "(+{best_updates.sum().div(best_updates.numel()):.2%})",
                )

        # unscale inputs and return
        return self.minmaxscale(x + best_p, False)

    def carlini(self, x, curr_p, prev_p, y):
        """
        This method serves as the hyperparameter criterion as used in
        https://arxiv.org/pdf/1608.04644.pdf. Specifically, it enforces the
        following scheme: if the current iterate is misclassifed and has lower
        norm than the previous iterate, then decrease the hyperparameter
        emphasizing misclassification, otherwise increase it.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param curr_p: the current batch of perturbation vectors
        :type curr_p: torch Tensor object (n, m)
        :param prev_p: the previous  batch of perturbation vectors
        :type prev_p: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: whether the hyperparameter should be decreased
        :rtype: torch Tensor object (n,)
        """
        return self.model.predict(x + curr_p).not_eq_(y) and curr_p.norm(
            self.norm, 1
        ) < prev_p.norm(self.norm, 1)

    def max_loss(self, x, curr_p, best_p, y):
        """
        This method serves as a restart criterion as used in
        https://arxiv.org/pdf/1706.06083.pdf. Specifically, it returns the set
        of perturbations whose loss is maximal.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param curr_p: the current batch of perturbation vectors
        :type curr_p: torch Tensor object (n, m)
        :param best_p: the previous  batch of perturbation vectors
        :type best_p: torch Tensor object (n, m )
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        return self.loss(self.model(x + curr_p), y).gt(
            self.loss(self.model(x + best_p)), y
        )

    def min_norm(self, x, curr_p, best_p, y):
        """
        This method serves as a restart criterion as used in
        https://arxiv.org/pdf/1907.02044.pdf. Specifically, it returns the set
        of perturbations whose norm is minimal (while still being
        misclassified).

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param curr_p: the current batch of perturbation vectors
        :type curr_p: torch Tensor object (n, m)
        :param best_p: the previous  batch of perturbation vectors
        :type best_p: torch Tensor object (n, m )
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        return self.model.predict(x + curr_p).not_eq_(y) and (
            curr_p.norm(self.norm, 1) < best_p.norm(self.norm, 1)
            or best_p.norm(self.norm, 1).eq(0)
        )

    def minmax_scale(self, x, transform=True):
        """
        This method serves as a wrapper for sklearn.preprocessing.MinMaxScaler.
        Specifically, it maps inputs to [0, 1] (as some techniques assume this
        range, e.g., change of variables). Importantly, since these scalers
        always return numpy arrays, this method additionally casts these inputs
        back as PyTorch tensors with the original data type.

        :param x: the batch of inputs to scale
        :type x: torch Tensor object (n, m)
        :param transform: performs the transformation if true; the inverse if false
        :return: a batch of scaled inputs
        :rtype: torch Tensor object (n, m)
        """
        if transform:
            self.scaler = sklearn.preprocessing.MinMaxScaler()
            self.scaler.dtype = x.dtype
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.inverse_transform(x)
        return torch.from_numpy(x).to(self.scaler.dtype)


class Attack:
    """
    The Attack class serves as a binder between Traveler and Surface objects
    with a high-level interface. Detailed in
    https://arxiv.org/pdf/2209.04521.pdf, attacks are built from a
    differentiable function (i.e., a surface) and routines to manipulate inputs
    (i.e., a traveler). Upon instantiation, the `craft` method serves as the
    main entry point in crafting adversarial examples.

    :func:`__init__`: instantiates Attack objects
    :func:`__repr__`: returns the attack name (based on the components)
    :func:`craft`: returns a batch of adversarial examples
    :func:`statistics`: prep misc statistics to be printed
    """

    def __init__(
        self,
        alpha,
        change_of_variables,
        clip,
        epochs,
        epsilon,
        loss_func,
        norm,
        model,
        optimizer_alg,
        random_restart,
        saliency_map,
        batch_size=-1,
        verbosity=0.25,
    ):
        """
        This method instantiates an Attack object with a variety of parameters
        necessary for building and coupling Traveler and Surface objects. The
        following parameters define high-level bookkeeping parameters across
        attacks:

        :param batch_size: crafting batch size (-1 for 1 batch)
        :type batch_size: int
        :param clip: range of allowable values for the domain
        :type clip: tuple of floats or tuple of torch Tensor object (n, m)
        :param epochs: number of optimization steps to perform
        :type epochs: int
        :param epsilon: lp-norm ball threat model
        :type epsilon: float
        :param verbosity: print attack statistics every verbosity%
        :type verbosity: float

        These subsequent parameters define the components of a Traveler object:

        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: bool
        :param optimizer_alg: optimization algorithm to use
        :type optimizer_alg: optimizer module class
        :param random_restart: whether to randomly perturb inputs
        :type random_restart: bool

        Finally, the following parameters define Surface objects:

        :param loss_func: objective function to differentiate
        :type loss_func: loss module class
        :param norm: lp-space to project gradients into
        :type norm: surface module callable
        :param model: neural network
        :type model: scikit-torch LinearClassifier-inherited object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: saliency module class

        To easily identify attacks, __repr__ is overriden and instead returns
        an abbreviation computed by concatenating the first letter (or two, if
        there is a name collision) of the following parameters, in order: (1)
        optimizer, (2) if random restart was used, (3) if change of variables
        was applied, (4) loss function, (5) saliency map, and (6) norm used.
        Combinations that yield known attacks are labeled as such (see __repr__
        for more details).

        :return: an attack
        :rtype: Attack object
        """

        # save attack parameters, and build short attack name
        self.batch_size = batch_size
        self.change_of_variables = change_of_variables
        self.clip = (-torch.inf, torch.inf) if change_of_variables else clip
        self.epochs = epochs
        self.epsilon = epsilon
        self.verbosity = max(1, int(epochs * verbosity))
        self.results = {m: [] for m in ("acc", "aloss", "mloss", "l0", "l2", "linf")}
        self.components = {
            "change of variables": change_of_variables,
            "loss function": loss_func.__name__,
            "optimizer": optimizer_alg.__name__,
            "random restart": random_restart,
            "saliency map": saliency_map.__name__,
            "target norm": "l∞" if norm == surface.linf else norm.__name__,
        }
        name = (
            self.components["optimizer"][0],
            "R" if self.components["random restart"] else "r̶",
            "V" if self.components["change of variables"] else "v̶",
            self.components["loss function"][:2],
            self.components["saliency map"][0],
            self.components["target norm"][1],
        )
        name_map = {
            ("M", "R", "v̶", "CE", "I", "∞"): "APGD-CE",
            ("M", "R", "v̶", "DL", "I", "∞"): "APGD-DLR",
            ("S", "r̶", "v̶", "CE", "I", "∞"): "BIM",
            ("A", "r̶", "V", "CW", "I", "2"): "CW-L2",
            ("S", "r̶", "v̶", "Id", "D", "2"): "DF",
            ("B", "r̶", "v̶", "Id", "D", "2"): "FAB",
            ("S", "R", "v̶", "CE", "I", "∞"): "PGD",
            ("S", "r̶", "v̶", "Id", "J", "0"): "JSMA",
        }
        self.name = name_map.get(name, "-".join(name))
        self.params = {"α": alpha, "ε": epsilon, "epochs": epochs}

        # instantiate traveler, surface, and necessary subcomponents
        norm_map = {surface.l0: 0, surface.linf: 1, surface.l2: 2}
        saliency_map = (
            saliency_map(norm_map[norm])
            if saliency_map is saliency.DeepFoolSaliency
            else saliency_map()
        )
        loss_func = loss_func()
        custom_opt_params = {
            "atk_loss": loss_func,
            "epochs": epochs,
            "epsilon": self.epsilon,
            "model": model,
        }
        torch_opt_params = {
            "lr": alpha,
            "maximize": loss_func.max_obj,
            "params": (torch.zeros(1),),
        }
        optimizer_alg = optimizer_alg(
            **(custom_opt_params | torch_opt_params)
            if optimizer_alg.__module__ == optimizer.__name__
            else torch_opt_params
        )
        self.traveler = traveler.Traveler(
            change_of_variables, optimizer_alg, random_restart * epsilon
        )
        self.surface = surface.Surface(
            loss_func, model, norm, saliency_map, change_of_variables
        )

        # collect any registered hyperparameters
        self.hparams = self.traveler.hparams | self.surface.hparams
        return None

    def __repr__(self):
        """
        This method returns a concise string representation of attack
        components and parameters. Notably, if the collection of components
        defines an attack made popular in the literature, it's full name is
        returned instead. The following named attacks are supported:

            APGD-CE (Auto-PGD with CE loss) (https://arxiv.org/pdf/2003.01690.pdf)
            APGD-DLR (Auto-PGD with DLR loss) (https://arxiv.org/pdf/2003.01690.pdf)
            BIM (Basic Iterative Method) (https://arxiv.org/pdf/1611.01236.pdf)
            CW-L2 (Carlini-Wagner with l₂ norm) (https://arxiv.org/pdf/1608.04644.pdf)
            DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
            FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
            JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)
            PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)

        :return: the attack name with parameters
        :rtype: str
        """
        return f"{self.name}({', '.join(f'{p}={v}' for p, v in self.params.items())})"

    def craft(self, x, y):
        """
        This method crafts adversarial examples, as defined by the attack
        parameters and the instantiated Travler and Surface attribute objects.
        Specifically, it normalizes inputs to be within [0, 1] (as some
        techniques assume this range, e.g., change of variables), creates a
        copy of x, creates the desired batch size, performs traveler
        initilizations (i.e., change of variables and random restart), and
        iterates an epoch number of times.

        :param x: the batch of inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: the batch of perturbation vectors
        :rtype: torch Tensor object (n, m)
        """

        # clone inputs, setup perturbation vector, chunks, epsilon, and clip
        org_x = x.detach()
        x = org_x.clone()
        p = x.new_zeros(x.size())
        batch_size = x.size(0) if self.batch_size == -1 else self.batch_size
        num_batches = -(-x.size(0) // batch_size)
        epsilon = (-self.epsilon, self.epsilon)
        clip = (
            self.clip
            if all(isinstance(c, torch.Tensor) for c in self.clip)
            else tuple(torch.full_like(x, c) for c in self.clip)
        )

        # craft adversarial examples per batch
        print(f"Crafting {x.size(0)} adversarial examples with {self}...")
        xi, yi, pi, cni, cxi = (t.split(batch_size) for t in (x, y, p, *clip))
        for b, (xb, yb, pb, cnb, cxb) in enumerate(zip(xi, yi, pi, cni, cxi), start=1):
            print(f"On batch {b} of {num_batches} {b/num_batches:.1%}...")
            min_p, max_p = (c.sub(xb).clamp(*epsilon) for c in (cnb, cxb))
            self.surface.initialize((min_p, max_p), pb)
            self.traveler.initialize(xb, pb)
            pb.clamp_(min_p, max_p)
            for epoch in range(self.epochs):
                self.surface(xb, yb, pb)
                self.traveler()
                pb.clamp_(min_p, max_p)
                print(
                    f"Epoch {epoch + 1:{len(str(self.epochs))}}/{self.epochs}",
                    self.progress(epoch, xb, yb, pb),
                ) if not (epoch + 1) % self.verbosity else print(
                    f"Epoch {epoch}... ({epoch/self.epochs:.1%})", end="\r"
                )

        # compute final perturbation if using cov and average statistics
        p = traveler.tanh_space(x + p).sub(org_x) if self.change_of_variables else p
        for m in ("acc", "l0", "l2", "linf"):
            self.results[m] = [s / y.numel() for s in self.results[m]]
        return p

    def progress(self, epoch, xb, yb, pb):
        """
        This method records various statistics on the adversarial example
        crafting process and returns a formatted string concisely representing
        the state of the attack. pecifically, this computes the following at
        each epoch (and the change since the last epoch): (1) model accuracy,
        (2) model loss, (3) attack loss, (3) l0, l2, and l∞ norms.

        :param epoch: current epoch
        :type epoch: int
        :param xb: current batch of inputs
        :type xb: torch Tensor object (n, m)
        :param yb: current batch of labels
        :type yb: torch Tensor object (n,)
        :param pb: current batch of perturbations
        :type pb: torch Tensor object (n, m)
        :return: print-ready statistics
        :rtype: str
        """

        # compute absolute statistics (averaged at the end)
        logits = self.surface.model(traveler.tanh_space(xb + pb), False)
        acc = logits.argmax(1).eq(yb).sum().item()
        aloss = self.surface.loss(logits, yb).sum().item()
        mloss = self.surface.model.loss(logits, yb).item()
        norms = [pb.norm(n, 1).sum().item() for n in (0, 2, torch.inf)]
        for m, s in zip(self.results.keys(), (acc, aloss, mloss, *norms)):
            try:
                self.results[m][epoch] += s
            except IndexError:
                self.results[m].append(s)

        # build str representation and return
        return (
            f"Accuracy: {acc / yb.numel():.1%} "
            f"({(acc - sum(self.results['acc'][-2:-1])) / yb.numel():+.1%}) "
            f"Model Loss: {mloss:.2f} "
            f"({mloss - sum(self.results['mloss'][-2:-1]):+6.2f}) "
            f"{self.name} Loss: {aloss:.2f} "
            f"({aloss - sum(self.results['aloss'][-2:-1]):+6.2f}) "
            f"l0: {norms[0] / yb.numel():.2f} "
            f"({(norms[0] - sum(self.results['l0'][-2:-1])) / yb.numel():+.2f}) "
            f"l2: {norms[1] / yb.numel():.2f} "
            f"({(norms[1] - sum(self.results['l2'][-2:-1])) / yb.numel():+.2f}) "
            f"l∞: {norms[2] / yb.numel():.2f} "
            f"({(norms[2] - sum(self.results['linf'][-2:-1])) / yb.numel():+.2f})"
        )


def attack_builder(
    alpha=None,
    clip=None,
    epochs=None,
    epsilon=None,
    model=None,
    change_of_variables_enabled=(True, False),
    optimizers=(
        optimizer.Adam,
        optimizer.BackwardSGD,
        optimizer.MomentumBestStart,
        optimizer.SGD,
    ),
    random_restart_enabled=(True, False),
    losses=(loss.CELoss, loss.CWLoss, loss.DLRLoss, loss.IdentityLoss),
    norms=(surface.l0, surface.l2, surface.linf),
    saliency_maps=(
        saliency.DeepFoolSaliency,
        saliency.JacobianSaliency,
        saliency.IdentitySaliency,
    ),
    verbosity=1,
):
    """
    As shown in https://arxiv.org/pdf/2209.04521.pdf, seminal attacks in
    machine learning can be cast into a single, unified framework. With this
    observation, this method combinatorically builds attack objects by swapping
    popular optimizers, norms, saliency maps, loss functions, and other
    techniques used within the AML community, such as random restart and change
    of variables. The combinations of supported components are shown below:

        Traveler Components:
        Change of variables: Enabled or disabled
        Optimizer: Adam, Backward Stochastic Gradient Descent,
                    Momentum Best Start, and Stochastic Gradient Descent
        Random restart: Enabled or disabled

        Surface Components:
        Loss: Categorical Cross-Entropy, Difference of Logits Ratio, Identity,
                and Carlini-Wagner
        Norm: l₀, l₂, and l∞
        Saliency map: DeepFool, Jacobian, and Identity

    We expose these supported components as arguments above for ease of
    instantiating a subset of the total combination space. Moreover, we also
    expose the following experimental parameters below to compare attacks.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: permissible values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :type model: scikit-torch LinearClassifier-inherited object
    :return: a generator yielding attack combinations
    :rtype: generator of Attack objects
    """

    # generate combinations of components and instantiate Attack objects
    num_attacks = (
        len(optimizers)
        * len(random_restart_enabled)
        * len(change_of_variables_enabled)
        * len(saliency_maps)
        * len(norms)
        * len(losses)
    )
    print(f"Yielding {num_attacks} attacks...")
    for (
        change_of_variables,
        loss_func,
        norm,
        optimizer_alg,
        random_restart,
        saliency_map,
    ) in itertools.product(
        change_of_variables_enabled,
        losses,
        norms,
        optimizers,
        random_restart_enabled,
        saliency_maps,
    ):
        yield Attack(
            alpha=alpha,
            change_of_variables=change_of_variables,
            epochs=epochs,
            loss=loss_func,
            model=model,
            norm=norm,
            optimizer=optimizer_alg,
            random_restart=random_restart,
            saliency_map=saliency_map,
            verbosity=verbosity,
        )


def apgdce(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Auto-PGD with Cross-Entropy loss
    (APGD-CE), as shown in https://arxiv.org/abs/2003.01690. Specifically,
    APGD-CE: does not use change of variables, uses the Momentum Best Start
    optimizer, uses random restart, uses Cross-Entropy loss, uses l∞ norm, and
    uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: APGD-CE attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer_alg=optimizer.MomentumBestStart,
        random_restart=True,
        loss_func=loss.CELoss,
        norm=surface.linf,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def apgddlr(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Auto-PGD with the Difference of
    Logits Ratio loss (APGD-DLR), as shown in
    https://arxiv.org/abs/2003.01690. Specifically, APGD-DLR: does not use
    change of variables, uses the Momentum Best Start optimizer, uses random
    restart, uses Difference of Logits ratio loss, uses l∞ norm, and uses the
    Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: APGD-DLR attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.DLRLoss,
        model=model,
        norm=surface.linf,
        optimizer_alg=optimizer.MomentumBestStart,
        random_restart=True,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def bim(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build the Basic Iterative Method (BIM),
    as shown in (https://arxiv.org/pdf/1611.01236.pdf) Specifically, BIM: does
    not use change of variables, uses the Stochastic Gradient Descent
    optimizer, does not use random restart, uses Cross Entropyy loss, uses l∞
    norm, and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: BIM attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CELoss,
        model=model,
        norm=surface.linf,
        optimizer_alg=optimizer.SGD,
        random_restart=False,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def cwl2(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Carlini-Wagner l₂ (CW-L2), as
    shown in (https://arxiv.org/pdf/1608.04644.pdf) Specifically, CW-L2: uses
    change of variables, uses the Adam optimizer, does not use random restart,
    uses Carlini-Wagner loss, uses l₂ norm, and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: Carlini-Wagner l₂ attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=True,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CWLoss,
        model=model,
        norm=surface.l2,
        optimizer_alg=optimizer.Adam,
        random_restart=False,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def df(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build DeepFool (DF), as shown in
    (https://arxiv.org/pdf/1511.04599.pdf) Specifically, DF: does not use
    change of variables, uses the Stochastic Gradient Descent optimizer, does
    not use random restart, uses Identity loss, uses l₂ norm, and uses the
    DeepFool saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.l2,
        optimizer_alg=optimizer.SGD,
        random_restart=False,
        saliency_map=saliency.DeepFoolSaliency,
        verbosity=verbosity,
    )


def fab(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Fast Adaptive Boundary (FAB), as
    shown in (https://arxiv.org/pdf/1907.02044.pdf) Specifically, FAB: does not
    use change of variables, uses the Backward Stochastic Gradient Descent
    optimizer, does not use random restart, uses Identity loss, uses l₂ norm,
    and uses the DeepFool saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.l2,
        optimizer_alg=optimizer.BackwardSGD,
        random_restart=False,
        saliency_map=saliency.DeepFoolSaliency,
        verbosity=verbosity,
    )


def pgd(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Projected Gradient Descent (PGD),
    as shown in (https://arxiv.org/pdf/1706.06083.pdf) Specifically, PGD: does
    not use change of variables, uses the Stochastic Gradient Descent
    optimizer, uses random restart, uses Identity loss, uses l∞ norm, and uses
    the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        clip=clip,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CELoss,
        model=model,
        norm=surface.linf,
        optimizer_alg=optimizer.SGD,
        random_restart=True,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def jsma(alpha=None, clip=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build the Jacobian-based Saliency Map
    Approach (JSMA), as shown in (https://arxiv.org/pdf/1511.07528.pdf)
    Specifically, the JSMA: does not use change of variables, uses the
    Stochastic Gradient Descent optimizer, does not use random restart, uses
    Identity loss, uses l0 norm, and uses the Jacobian saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param clip: range of allowable values for the domain
    :type clip: tuple of floats or torch Tensor object (n, m)
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: scikit-torch LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        clip=clip,
        epochs=epochs,
        change_of_variables=False,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.l0,
        optimizer_alg=optimizer.SGD,
        random_restart=False,
        saliency_map=saliency.JacobianSaliency,
        verbosity=verbosity,
    )


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
