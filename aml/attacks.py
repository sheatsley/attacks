"""
This module defines the attacks proposed in
https://arxiv.org/pdf/2209.04521.pdf.
Authors: Ryan Sheatsley & Blaine Hoak
Mon Apr 18 2022
"""
import aml.loss as loss  # Loss functions for crafting adversarial examples in PyTorch
import aml.optimizer as optimizer  # Optimizers for crafting adversarial examples in PyTorch
import aml.surface as surface  # Models for crafting adversarial examples
import aml.traveler as traveler  # Perturbation schemes for adversarial examples
import itertools  # Functions creating iterators for efficietn looping
import pandas  # Python Data Analysis Library
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class Adversary:
    """
    The Adversary class supports generalized threat modeling. Specifically, as
    shown in https://arxiv.org/pdf/1608.04644.pdf and
    https://arxiv.org/pdf/1907.02044.pdf, effective adversaries often enact
    some decision after adversarial examples have been crafted. For the
    provided examples, such adversaries embed hyperparamter optimization,
    perform multiple restarts (for non-deterministic attacks), and return the
    most effective adversarial examples. This class provides such functions.

    :func:`__init__`: instantiates Adversary objects
    :func:`__getattr__`: grab Attack object attributes
    :func:`__repr__`: returns the threat model
    :func:`binary_search`: optimizes hyperparamters via binary search
    :func:`craft`: returns a set of adversarial examples
    :func:`max_loss`: restart rule used in PGD attack
    :func:`min_norm`: restart rule used in FAB attack
    :func:`misclassified`: hyperparameter rule used in CW-L2 attack
    """

    def __init__(
        self,
        best_update,
        hparam,
        hparam_bounds,
        hparam_steps,
        hparam_update,
        num_restarts,
        verbose=False,
        **atk_args,
    ):
        """
        This method instantiates an Adversary object with a hyperparameter to
        be optimized (via binary search), the lower and upper bounds to
        consider for binary search, the update rule to control the
        hyperparameter search (by convention, "success" is encoded as True,
        meaning the hyperparameter will be decreased for the next iteration),
        the number of hyperparameter optimization steps, the number of
        restarts, and the update rule to determine if an adversarial example is
        kept across restarts.

        :param best_update: update rule for the best adversarial examples
        :type best_update: Adversary class method or None
        :param hparam: the hyperparameter to be optimized
        :type hparam: str or None
        :param hparam_bounds: lower and upper bounds for binary search
        :type hparam_bounds: tuple of floats or None
        :param hparam_steps: the number of hyperparameter optimization steps
        :type hparam_steps: int or None
        :param hparam_update: update rule for hyperparameter optimization
        :type hparam_update: Adversary class method or None
        :param num_restarts: number of restarts to consider
        :type num_restarts: int or None
        :param verbose: whether progress statistics are printed
        :type verbose: bool
        :param atk_args: attack parameters
        :type atk_args: dict
        :return: an adversary
        :rtype: Adversary object
        """

        # instantiate attack and configure arguments
        attack = Attack(**atk_args)
        best_update = type(None) if best_update is None else best_update.__get__(self)
        hparam_update = (
            type(None) if hparam_update is None else hparam_update.__get__(self)
        )
        num_restarts = 0 if num_restarts is None else num_restarts

        # set attributes
        self.craft_p = (
            (lambda x, y, b: (attack.craft(x, y, b),))
            if hparam is None
            else self.binary_search
        )
        self.best_update = best_update
        self.hparam = hparam
        self.hparam_bounds = hparam_bounds
        self.hparam_steps = hparam_steps
        self.hparam_update = hparam_update
        self.loss = attack.surface.loss
        self.model = attack.surface.model
        self.name = attack.name
        self.num_restarts = num_restarts + 1
        self.verbose = verbose
        self.params = {
            "best_update": best_update.__name__,
            "hparam": hparam,
            "hparam_bounds": hparam_bounds,
            "hparam_update": hparam_update.__name__,
            "hparam_steps": hparam_steps,
            "num_restarts": num_restarts,
            "attack": attack,
        }
        return None

    def __getattr__(self, name):
        """
        This method aliases Attack objects (i.e., self.attack) attributes to be
        accessible by this object directly. It is principally used for better
        readability and easier debugging.

        :param name: name of the attribute to recieve from self.attack
        :type name: str
        :return: the desired attribute (if it exists)
        :rtype: misc
        """
        return self.__getattribute__("params")["attack"].__getattribute__(name)

    def __repr__(self):
        """
        This method returns a concise string representation of adversary
        and attack parameters.

        :return: adversarial and attack parameters
        :rtype: str
        """
        return f"Adversary({', '.join(f'{p}={v}' for p, v in self.params.items())})"

    def binary_search(self, x, y, b):
        """
        This method performs binary search on a hyperparameter, such as c used
        in Carlini-Wagner loss (https://arxiv.org/pdf/1608.04644.pdf). By
        convention, success (encoded as True by any update rule) decreases the
        hyperparameter, while failure (encoded as False) will increase it.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :return: adversarial perturbations
        :rtype: generator of torch Tensor objects (n, m)
        """
        lb, ub = torch.tensor(self.hparam_bounds).repeat(y.numel(), 1).unbind(1)
        for h in range(1, self.hparam_steps + 1):
            p = self.params["attack"].craft(x, y, b)
            self.verbose and print(
                f"On hyperparameter iteration {h} of {self.hparam_steps}...",
                f"({h / self.hparam_steps:.1%})",
            )
            success = self.hparam_update(x, p, y)
            hparam = self.params["attack"].hparams[self.hparam]
            ub[success] = ub[success].minimum(hparam[success])
            lb[~success] = lb[~success].maximum(hparam[~success])
            hparam.copy_(ub.add(lb).div(2))
            self.verbose and print(
                f"Hyperparameter {self.hparam} updated.",
                f"{success.sum().div(success.numel()):.2%} success",
            )
            yield p

    def craft(self, x, y):
        """
        This method represents the core of the Adversary class. Specifically,
        this class produces the most effective adversarial examples,
        parameterized by restarts and hyperparameter optimization.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :return: adversarial perturbations
        :rtype: torch Tensor object (n, m)
        """

        # instantiate bookkeeping and iterate over restarts & hyperparameters
        x = x.clone()
        b = torch.full_like(x, torch.inf)
        b[self.model(x).argmax(1).ne(y)] = 0
        for r in range(self.num_restarts):
            r > 0 and self.verbose and print(
                f"On restart iteration {r} of {self.num_restarts - 1}...",
                f"({r / (self.num_restarts - 1):.1%})",
            )

            # store the best adversarial perturbations seen thus far
            for p in self.craft_p(x, y, b):
                update = self.best_update(x, p, b, y)
                non_adv = b.isinf().any(1)
                b[update] = p[update]
                new = update.logical_and(non_adv).sum().div(update.numel())
                improved = update.logical_and(~non_adv).sum().div(update.numel())
                self.verbose and print(
                    f"Found {update.sum()} better adversarial examples!",
                    f"(+{update.sum().div(update.numel()):.2%})",
                    f"(New {new:.2%}, Improved {improved:.2%})",
                )
        return b.nan_to_num_(nan=None, posinf=0)

    def max_loss(self, x, p, b, y):
        """
        This method serves as a restart rule as used in
        https://arxiv.org/pdf/1706.06083.pdf. Specifically, it returns the set
        of perturbations whose loss improved.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param p: candidate set of perturbation vectors
        :type p: torch Tensor object (n, m)
        :param b: best set of perturbation vectors
        :type b: torch Tensor object (n, m )
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        ploss = self.loss(self.model(x + p), y)
        bloss = self.loss(self.model(x + b.nan_to_num(posinf=0)), y)
        return ploss.gt(bloss) if self.loss.max_obj else ploss.lt(bloss)

    def min_norm(self, x, p, b, y):
        """
        This method serves as a restart rule as used in
        https://arxiv.org/pdf/1907.02044.pdf and
        https://arxiv.org/pdf/1608.04644.pdf. Specifically, it returns the set
        of misclassified perturbations with minimal norm.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param p: candidate set of perturbation vectors
        :type p: torch Tensor object (n, m)
        :param b: best set of perturbation vectors
        :type b: torch Tensor object (n, m )
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        return self.misclassified(x, p, y).logical_and_(
            p.norm(self.lp, 1) < b.norm(self.lp, 1)
        )

    def misclassified(self, x, p, y):
        """
        This method serves as a hyperparameter rule as used in
        https://arxiv.org/pdf/1608.04644.pdf. Specifically, it returns the set
        of adversarial examples that are misclassified.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param p: set of perturbation vectors
        :type p: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        return self.model(x + p).argmax(1).ne(y)


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
    :func:`craft`: returns a set of adversarial examples
    :func:`progress`: records various statistics on crafting progress
    :func:`update`: updates output adversarial perturbations
    """

    def __init__(
        self,
        alpha,
        early_termination,
        epochs,
        epsilon,
        loss_func,
        norm,
        model,
        optimizer_alg,
        random_start,
        saliency_map,
        batch_size=-1,
        statistics=False,
        verbosity=0.25,
    ):
        """
        This method instantiates an Attack object with a variety of parameters
        necessary for building and coupling Traveler and Surface objects. The
        following parameters define high-level bookkeeping parameters across
        attacks:

        :param batch_size: crafting batch size (-1 for 1 batch)
        :type batch_size: int
        :param early_termination: whether misclassified inputs are perturbed
        :type early_termination: bool
        :param epochs: number of optimization steps to perform
        :type epochs: int
        :param epsilon: lp-norm ball threat model
        :type epsilon: float
        :param statistics: save attack progress (heavily increases compute)
        :type statistics: bool
        :param verbosity: print attack statistics every verbosity%
        :type verbosity: float

        These subsequent parameters define the components of a Traveler object:

        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param optimizer_alg: optimization algorithm to use
        :type optimizer_alg: optimizer module class
        :param random_start: desired random start heuristic
        :type random_start: traveler module class

        Finally, the following parameters define Surface objects:

        :param loss_func: objective function to differentiate
        :type loss_func: loss module class
        :param norm: lp-space to project gradients and enforce threat models
        :type norm: surface module class
        :param model: neural network
        :type model: dlm LinearClassifier-inherited object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: surface module class

        To easily identify attacks, __repr__ is overriden and instead returns
        an abbreviation computed by concatenating the first letter (or two, if
        there is a name collision) of the following parameters, in order: (1)
        optimizer, (2) random start strategy, (3) loss function, (4) saliency
        map, and (5) norm used. Combinations that yield known attacks are
        labeled as such (see __repr__ for more details).

        :return: an attack
        :rtype: Attack object
        """

        # set & save attack parameters, and build short attack name
        self.batch_size = batch_size
        self.epochs = epochs
        self.epsilon = epsilon
        self.et = early_termination
        self.lp = {surface.L0: 0, surface.L2: 2, surface.Linf: torch.inf}[norm]
        self.statistics = statistics
        self.verbosity = max(int(epochs * verbosity), 1 if verbosity else 0)
        self.components = {
            "loss function": loss_func.__name__,
            "optimizer": optimizer_alg.__name__,
            "random start": random_start.__name__,
            "saliency map": saliency_map.__name__,
            "target norm": "l∞" if norm == surface.Linf else norm.__name__,
        }
        name = (
            self.components["optimizer"][0],
            self.components["random start"][0],
            self.components["loss function"][:2],
            self.components["saliency map"][0],
            self.components["target norm"][1],
        )
        name_map = {
            ("M", "M", "CE", "I", "∞"): "APGD-CE",
            ("M", "M", "DL", "I", "∞"): "APGD-DLR",
            ("S", "I", "CE", "I", "∞"): "BIM",
            ("A", "I", "CW", "I", "2"): "CW-L2",
            ("S", "I", "Id", "D", "2"): "DF",
            ("B", "S", "Id", "D", "2"): "FAB",
            ("S", "M", "CE", "I", "∞"): "PGD",
            ("S", "I", "Id", "J", "0"): "JSMA",
        }
        self.name = name_map.get(name, "-".join(name))
        self.params = {"α": alpha, "ε": epsilon, "epochs": epochs, "min dist": self.et}

        # instantiate traveler, surface, and necessary subcomponents
        norm = norm(epsilon=epsilon, maximize=loss_func.max_obj)
        self.project = norm.project
        saliency_map = saliency_map(classes=model.params["classes"], p=self.lp)
        loss_func = loss_func(classes=model.params["classes"])
        custom_opt_params = {
            "atk_loss": loss_func,
            "epochs": epochs,
            "epsilon": alpha if self.lp == 0 else epsilon,
            "model": model,
            "norm": self.lp,
            "smap": saliency_map,
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
        random_start = random_start(norm=self.lp, epsilon=epsilon)
        self.traveler = traveler.Traveler(optimizer_alg, random_start)
        self.surface = surface.Surface(loss_func, model, norm, saliency_map)

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
        CW-L2 (Carlini-Wagner with l2 norm) (https://arxiv.org/pdf/1608.04644.pdf)
        DF (DeepFool) (https://arxiv.org/pdf/1511.04599.pdf)
        FAB (Fast Adaptive Boundary) (https://arxiv.org/pdf/1907.02044.pdf)
        JSMA (Jacobian Saliency Map Approach) (https://arxiv.org/pdf/1511.07528.pdf)
        PGD (Projected Gradient Descent) (https://arxiv.org/pdf/1706.06083.pdf)

        :return: the attack name with parameters
        :rtype: str
        """
        return f"{self.name}({', '.join(f'{p}={v}' for p, v in self.params.items())})"

    def craft(self, x, y, o=None):
        """
        This method crafts adversarial examples, as defined by the attack
        parameters and the instantiated Traveler and Surface objects.
        Specifically, it creates a copy of x, batches inputs, optionally
        initializes the output buffer to a set of optimal perturbations (useful
        for attacks that perform shrinking start), performs surface & traveler
        initilizations, and finally iterates an epoch number of times.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param o: initialize output buffer to saved adversarial examples
        :type o: torch Tensor object (n, m)
        :return: set of perturbation vectors
        :rtype: torch Tensor object (n, m)
        """

        # initialize perturbations & ranges and compute batch sizes
        x = x.detach().clone()
        p = torch.zeros_like(x)
        mins, maxs = (x.min(0).values.clamp(max=0), x.max(0).values.clamp(min=1))
        batch_size = x.size(0) if self.batch_size == -1 else self.batch_size
        bmax = -(-x.size(0) // batch_size)

        # if needed, init output buffer & set misclassified perturbations to 0
        correct = self.surface.model(x).argmax(1).eq(y).unsqueeze_(1)
        o = torch.full_like(p, torch.inf).where(correct, p) if o is None else o.clone()

        # set verbosity and configure batch & output results dataframes
        verbose = self.verbosity and self.verbosity != self.epochs
        rows = [e for i in range(bmax) for e in range(self.epochs + 1)]
        metrics = "accuracy", "model_loss", "attack_loss", "l0", "l2", "linf"
        self.b_results = pandas.DataFrame(0, index=rows, columns=metrics)
        self.results = pandas.DataFrame(0, index=rows, columns=metrics)

        # configure batches, attach objects, & apply perturbation inits
        verbose and print(f"Crafting {x.size(0)} adversarial examples with {self}...")
        xi, yi, pi, oi = (t.split(batch_size) for t in (x, y, p, o))
        for b, (xb, yb, pb, ob) in enumerate(zip(xi, yi, pi, oi)):
            verbose and print(f"On batch {b + 1} of {bmax} {(b + 1) / bmax:.1%}...")
            cnb, cxb = mins.sub(xb), maxs.sub(xb)
            self.surface.initialize((cnb, cxb), pb)
            self.traveler.initialize(pb, ob)
            pb.clamp_(cnb, cxb)
            self.progress(b, 0, xb, yb, pb, ob)

            # compute peturbation updates and record progress
            for e in range(1, self.epochs + 1):
                self.surface(xb, yb, pb)
                self.traveler()
                pb.clamp_(cnb, cxb)
                None if self.et else self.project(pb)
                self.update(xb, yb, pb, ob)
                prog = self.progress(b, e, xb, yb, pb, ob) if self.statistics else ""
                print(
                    f"Epoch {e:{len(str(self.epochs))}} / {self.epochs} {prog:<15}"
                ) if verbose and not e % self.verbosity else print(
                    f"{self.name}: Epoch {e}... ({e / self.epochs:.1%})", end="\r"
                )

        # compute final statistics, set failed perturbations to zero and return
        self.b_results = self.b_results.groupby(self.b_results.index).sum()
        self.results = self.results.groupby(self.results.index).sum()
        self.b_results[["accuracy", "l0", "l2", "linf"]] /= y.numel()
        self.results[["accuracy", "l0", "l2", "linf"]] /= y.numel()
        return o.nan_to_num_(nan=None, posinf=0)

    def progress(self, batch, epoch, xb, yb, pb, ob):
        """
        This method measures various statistics on the adversarial crafting
        process & a formatted string concisely representing the state of the
        attack. Specifically, this computes the following at each epoch (and
        the change since the last epoch): (1) model accuracy, (2) model loss,
        (3) attack loss, (3) l0, l2, and l∞ norms. Notably, due to performance
        bugs in PyTorch (https://github.com/pytorch/pytorch/issues/51509) and
        the number of forward passes necessary to compute interesting
        statistics, this method can have a high performance impact.

        :param batch: current batch number
        :type batch: int
        :param epoch: current epoch within the batch
        :type epoch: int
        :param xb: current batch of inputs
        :type xb: torch Tensor object (n, m)
        :param yb: current batch of labels
        :type yb: torch Tensor object (n,)
        :param pb: current batch of perturbations
        :type pb: torch Tensor object (n, m)
        :param ob: final batch of perturbations (when using early termination)
        :type ob: torch Tensor object (n, m)
        :return: print-ready statistics
        :rtype: str
        """

        # compute current index and project onto lp-threat model
        idx = batch * (self.epochs + 1) + epoch
        norms = (0, 2, torch.inf)
        pb = self.project(pb.clone())

        # compute batch buffer stats and update results
        logits = self.surface.model(xb + pb)
        mloss = self.surface.model.loss(logits, yb).item()
        aloss = self.surface.loss(logits, yb).sum().item()
        acc = logits.argmax(1).eq_(yb).sum().item()
        nb = [pb.norm(n, 1).sum().item() for n in norms]
        self.b_results.iloc[idx] += (acc, mloss, aloss, *nb)

        # compute output buffer stats and update results
        ologits = self.surface.model(xb + ob.nan_to_num(posinf=0))
        omloss = self.surface.model.loss(ologits, yb).item()
        oaloss = self.surface.loss(logits, yb).sum().item()
        oacc = ologits.argmax(1).eq_(yb).sum().item()
        on = [ob.nan_to_num(posinf=0).norm(n, 1).sum().item() for n in norms]
        self.results.iloc[idx] += (oacc, omloss, oaloss, *on)

        # build str representation and return
        return (
            f"Output Acc: {oacc / yb.numel():.1%} Batch Acc: {acc / yb.numel():.1%} "
            f"({(acc - self.b_results.accuracy.iloc[idx - 1]) / yb.numel():+.1%}) "
            f"Model Loss: {mloss:.2f} "
            f"({(mloss - self.b_results.model_loss.iloc[idx - 1]):+6.2f}) "
            f"{self.name} Loss: {aloss:.2f} "
            f"({(aloss - self.b_results.attack_loss.iloc[idx - 1]):+6.2f}) "
            f"l0: {nb[0] / yb.numel():.2f} "
            f"({(nb[0] -  self.b_results.l0.iloc[idx - 1]) / yb.numel():+.2f}) "
            f"l2: {nb[1] / yb.numel():.2f} "
            f"({(nb[1] - self.b_results.l2.iloc[idx - 1]) / yb.numel():+.2f}) "
            f"l∞: {nb[2] / yb.numel():.2f} "
            f"({(nb[2] - self.b_results.linf.iloc[idx - 1]) / yb.numel():+.2f})"
        )

    def update(self, xb, yb, pb, ob):
        """
        This method updates the final perturbations returned by attacks.
        Specifically, the update is determined as: (1) for minimum-norm
        adversaries, perturbations that are both (a) misclassified, and (b) a
        smaller lp-norm than the current best perturbations are saved, while
        (2) for maximum-loss adversaries, perturbations that have greater model
        loss than the current best perturbations are saved.

        :param xb: current batch of inputs
        :type xb: torch Tensor object (n, m)
        :param yb: current batch of labels
        :type yb: torch Tensor object (n,)
        :param pb: current batch of perturbations
        :type pb: torch Tensor object (n, m)
        :param ob: final batch of perturbations (when using early termination)
        :type ob: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """
        pb = self.project(pb.clone())
        logits = self.surface.model(xb + pb)
        if self.et:
            correct = logits.argmax(1).eq(yb)
            smaller = pb.norm(self.lp, dim=1).lt(ob.norm(self.lp, dim=1))
            update = (~correct).logical_and_(smaller)
        else:
            ploss = self.surface.loss(logits, yb)
            ologits = self.surface.model(xb + ob.nan_to_num(posinf=0))
            oloss = self.surface.loss(ologits, yb)
            update = oloss.lt(ploss) if self.surface.loss.max_obj else oloss.gt(ploss)
        ob[update] = pb[update]
        return None


def attack_builder(
    alpha,
    early_termination,
    epochs,
    epsilon,
    model,
    losses=(loss.CWLoss, loss.CELoss, loss.DLRLoss, loss.IdentityLoss),
    norms=(surface.L0, surface.L2, surface.Linf),
    optimizers=(
        optimizer.Adam,
        optimizer.BackwardSGD,
        optimizer.MomentumBestStart,
        optimizer.SGD,
    ),
    random_starts=(
        traveler.IdentityStart,
        traveler.MaxStart,
        traveler.ShrinkingStart,
    ),
    saliency_maps=(
        surface.DeepFoolSaliency,
        surface.IdentitySaliency,
        surface.JacobianSaliency,
    ),
    statistics=False,
    verbosity=1,
):
    """
    As shown in https://arxiv.org/pdf/2209.04521.pdf, seminal attacks in
    machine learning can be cast into a single, unified framework. With this
    observation, this method combinatorically builds attack objects by swapping
    popular optimizers, random start strategies, norms, saliency maps, and loss
    functions. The combinations of supported components are shown below:

        Traveler Components:
        Optimizer: Adam, Backward Stochastic Gradient Descent,
                    Momentum Best Start, and Stochastic Gradient Descent
        Random start: Identity Start, Max Start, and Shrinking Start

        Surface Components:
        Loss: Carlini-Wagner, Categorical Cross-Entropy,
                Difference of Logits Ratio, and Identity
        Saliency map: DeepFool, Identity, and Jacobian
        Norm: l0, l2, and l∞

    We expose these supported components as arguments above for ease of
    instantiating a subset of the total combination space. Moreover, we also
    expose the following experimental parameters below to compare attacks.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model(s)
    :type epsilon: float or tuple of floats
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param losses: losses to consider
    :type losses: tuple of loss module classes
    :param norms: lp-norms to consider
    :type norms: tuple of Surface object methods
    :param optimizers: optimizers to consider
    :type optimizers: tuple of optimizer module classes
    :param random_starts: random start strategies to consider
    :type random_starts: tuple of traveler module classes
    :param saliency_maps: saliency maps to consider
    :type saliency_maps: tuple of surface module callables
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: a generator yielding attack combinations
    :rtype: generator of Attack objects
    """

    # create epsilon mapping
    ns = (surface.L0, surface.L2, surface.Linf)
    eps = epsilon if type(epsilon) is tuple else itertools.repeat(epsilon)
    lp = {n: e for n, e in zip(ns, eps)}

    # generate combinations of components and instantiate Attack objects
    num_attacks = (
        len(optimizers)
        * len(random_starts)
        * len(saliency_maps)
        * len(norms)
        * len(losses)
    )
    print(f"Yielding {num_attacks} attacks...")
    for (
        norm,
        saliency_map,
        loss_func,
        random_start,
        optimizer_alg,
    ) in itertools.product(
        norms,
        saliency_maps,
        losses,
        random_starts,
        optimizers,
    ):
        yield Attack(
            alpha=alpha,
            early_termination=early_termination,
            epochs=epochs,
            epsilon=lp[norm],
            loss_func=loss_func,
            model=model,
            norm=norm,
            optimizer_alg=optimizer_alg,
            random_start=random_start,
            saliency_map=saliency_map,
            statistics=statistics,
            verbosity=verbosity,
        )


def apgdce(
    alpha, epochs, epsilon, model, num_restarts=3, statistics=False, verbosity=1
):
    """
    This function serves as an alias to build Auto-PGD with Cross-Entropy loss
    (APGD-CE), as shown in https://arxiv.org/abs/2003.01690. Specifically,
    APGD-CE: uses the Momentum Best Start optimizer, uses Max Start random
    start, uses Cross-Entropy loss, uses l∞ norm, and uses the Identity
    saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of restarts to perform
    :type num_restarts: int
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: APGD-CE attack
    :rtype: Attack objects
    """
    return Adversary(
        best_update=Adversary.max_loss,
        hparam=None,
        hparam_bounds=None,
        hparam_update=None,
        hparam_steps=None,
        num_restarts=num_restarts,
        alpha=alpha,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        optimizer_alg=optimizer.MomentumBestStart,
        random_start=traveler.MaxStart,
        loss_func=loss.CELoss,
        norm=surface.Linf,
        saliency_map=surface.IdentitySaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def apgddlr(
    alpha, epochs, epsilon, model, num_restarts=3, statistics=False, verbosity=1
):
    """
    This function serves as an alias to build Auto-PGD with the Difference of
    Logits Ratio loss (APGD-DLR), as shown in https://arxiv.org/abs/2003.01690.
    Specifically, APGD-DLR: uses the Momentum Best Start optimizer, uses Max
    Start random start, uses Difference of Logits ratio loss, uses l∞ norm, and
    uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of restarts to perform
    :type num_restarts: int
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: APGD-DLR attack
    :rtype: Attack objects
    """
    return Adversary(
        best_update=Adversary.max_loss,
        hparam=None,
        hparam_bounds=None,
        hparam_update=None,
        hparam_steps=None,
        num_restarts=num_restarts,
        verbose=True if verbosity not in {0, 1} else False,
        alpha=alpha,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.DLRLoss,
        model=model,
        norm=surface.Linf,
        optimizer_alg=optimizer.MomentumBestStart,
        random_start=traveler.MaxStart,
        saliency_map=surface.IdentitySaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def bim(alpha, epochs, epsilon, model, statistics=False, verbosity=1):
    """
    This function serves as an alias to build the Basic Iterative Method (BIM),
    as shown in (https://arxiv.org/pdf/1611.01236.pdf) Specifically, BIM: uses
    the Stochastic Gradient Descent optimizer, uses Identity random start, uses
    Cross Entropyy loss, uses l∞ norm, and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: BIM attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CELoss,
        model=model,
        norm=surface.Linf,
        optimizer_alg=optimizer.SGD,
        random_start=traveler.IdentityStart,
        saliency_map=surface.IdentitySaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def cwl2(
    alpha,
    epochs,
    epsilon,
    model,
    hparam_bounds=(0, 1e10),
    hparam_steps=9,
    statistics=False,
    verbosity=1,
):
    """
    This function serves as an alias to build Carlini-Wagner l₂ (CW-L2), as
    shown in (https://arxiv.org/pdf/1608.04644.pdf) Specifically, CW-L2: uses
    the Adam optimizer, uses Identity random start, uses Carlini-Wagner loss,
    uses l2 norm, and uses the Identity saliency map. It also performs binary
    search over c, which controls the importance of inducing misclassification
    over minimizing norm.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param hparam_bounds: lower and upper bounds for binary search
    :type hparam_bounds: tuple of floats
    :param hparam_steps: the number of hyperparameter optimization steps
    :type hparam_steps: int
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: Carlini-Wagner l₂ attack
    :rtype: Attack objects
    """
    return Adversary(
        best_update=Adversary.min_norm,
        hparam="c",
        hparam_bounds=hparam_bounds,
        hparam_steps=hparam_steps,
        hparam_update=Adversary.misclassified,
        num_restarts=0,
        verbose=True if verbosity not in {0, 1} else False,
        alpha=alpha,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CWLoss,
        model=model,
        norm=surface.L2,
        optimizer_alg=optimizer.Adam,
        random_start=traveler.IdentityStart,
        saliency_map=surface.IdentitySaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def df(alpha, epochs, epsilon, model, statistics=True, verbosity=1):
    """
    This function serves as an alias to build DeepFool (DF), as shown in
    (https://arxiv.org/pdf/1511.04599.pdf) Specifically, DF: uses the
    Stochastic Gradient Descent optimizer, use Identity random start, uses
    Identity loss, uses l2 norm, and uses the DeepFool saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.L2,
        optimizer_alg=optimizer.SGD,
        random_start=traveler.IdentityStart,
        saliency_map=surface.DeepFoolSaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def fab(alpha, epochs, epsilon, model, num_restarts=2, statistics=False, verbosity=1):
    """
    This function serves as an alias to build Fast Adaptive Boundary (FAB), as
    shown in (https://arxiv.org/pdf/1907.02044.pdf) Specifically, FAB: uses the
    Backward Stochastic Gradient Descent optimizer, uses Shrinking Start random
    start, uses Identity loss, uses l2 norm, and uses the DeepFool saliency
    map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of restarts to perform
    :type num_restarts: int
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Adversary(
        best_update=Adversary.min_norm,
        hparam=None,
        hparam_bounds=None,
        hparam_update=None,
        hparam_steps=None,
        num_restarts=num_restarts,
        verbose=True if verbosity not in {0, 1} else False,
        alpha=alpha,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.L2,
        optimizer_alg=optimizer.BackwardSGD,
        random_start=traveler.ShrinkingStart,
        saliency_map=surface.DeepFoolSaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def jsma(alpha, epochs, epsilon, model, statistics=False, verbosity=1):
    """
    This function serves as an alias to build the Jacobian-based Saliency Map
    Approach (JSMA), as shown in (https://arxiv.org/pdf/1511.07528.pdf)
    Specifically, the JSMA: uses the Stochastic Gradient Descent optimizer,
    uses Identity random start, uses Identity loss, uses l0 norm, and uses the
    Jacobian saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.L0,
        optimizer_alg=optimizer.SGD,
        random_start=traveler.IdentityStart,
        saliency_map=surface.JacobianSaliency,
        statistics=statistics,
        verbosity=verbosity,
    )


def pgd(alpha, epochs, epsilon, model, statistics=False, verbosity=1):
    """
    This function serves as an alias to build Projected Gradient Descent (PGD),
    as shown in (https://arxiv.org/pdf/1706.06083.pdf) Specifically, PGD: uses
    the Stochastic Gradient Descent optimizer, uses Max Start random start,
    uses Identity loss, uses l∞ norm, and uses the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CELoss,
        model=model,
        norm=surface.Linf,
        optimizer_alg=optimizer.SGD,
        random_start=traveler.MaxStart,
        saliency_map=surface.IdentitySaliency,
        statistics=statistics,
        verbosity=verbosity,
    )
