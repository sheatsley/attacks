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
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# finish converting attacks to adversaries
# add perturbations visualizations in examples directory
# l2 rr should be normlized by l2-norm and l0 norm should pick max l0 random features
# for l0 clamp, need to check if cov makes 0s a very small number (check on prev_p == 0 would fail)
# consider setting a min in tanh_p (like l2 norm) to mitigate underflow
# configure adversary verbosity based on attack verbosity (use \r if no attack prints)
# consider showing separate batch and output accuracies (attacks like fab almost always show 50% acc)
# probably seprate out essential from non-essential ops in progress
# confirm progress logic correcetness for max-loss cov
# consider if hparam "success" flag is needed to determine if hparams should be increased or decreased
# consider dropping attack alias getattr... probably more harm than good


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
    :func:`__repr__`: returns the threat model
    :func:`attack`: returns a set of adversarial examples
    :func:`max_loss`: restart criterion used in PGD attack
    :func:`min_norm`: restart criterion used in FAB attack
    :func:`misclassified`: hyperparameter criterion used in CW-L2 attack
    """

    def __init__(
        self,
        hparam,
        hparam_bounds,
        hparam_criterion,
        hparam_steps,
        num_restarts,
        restart_criterion,
        verbose=False,
        **atk_args,
    ):
        """
        This method instantiates an Adversary object with a hyperparameter to
        be optimized (via binary search), the lower and upper bounds to
        consider for binary search, the criterion to control the hyperparameter
        search (by convention, "success" is encoded as True, meaning the
        hyperparameter will be decreased for the next iteration), the number of
        hyperparameter optimization steps, the number of restarts, and the
        criterion to determine if an input is kept across restarts.

        :param hparam: the hyperparameter to be optimized
        :type hparam: str
        :param hparam_bounds: lower and upper bounds for binary search
        :type hparam_bounds: tuple of floats
        :param hparam_criterion: criterion for hyperparameter optimization
        :type hparam_criterion: callable
        :param hparam_steps: the number of hyperparameter optimization steps
        :type hparam_steps: int
        :param num_restarts: number of restarts to consider
        :type num_restarts: int
        :param restart_criterion: criterion for keeping inputs across restarts
        :type restart_criterion: callable
        :param verbose: whether progress statistics are printed
        :type verbose: bool
        :param atk_args: attack parameters
        :type atk_args: dict
        :return: an adversary
        :rtype: Adversary object
        """
        attack = Attack(**atk_args)
        self.hparam = hparam
        self.hparam_bounds = hparam_bounds
        self.hparam_criterion = hparam_criterion
        self.hparam_steps = max(hparam_steps, 1)
        self.num_restarts = max(num_restarts, 1)
        self.restart_criterion = restart_criterion
        self.attack_alg = attack
        self.loss = atk_args["loss_func"]
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
        return self.__getattribute__("attack_alg").__getattribute__(name)

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

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: adversarial examples
        :rtype: torch Tensor object (n, m)
        """

        # iterate over restarts & hyperparameters
        b = torch.full_like(x, torch.inf)
        for r in range(1, self.num_restarts + 1):
            print(
                f"On restart iteration {r} of {self.num_restarts}...",
                f"({r / self.num_restarts:.1%})",
            )

            # instantiate bookkeeping and apply hyperparameter criterion
            p = torch.zeros_like(x)
            lb, ub = torch.tensor(self.hparam_bounds).repeat(y.numel(), 1).unbind(1)
            for h in range(1, self.hparam_steps + 1):
                print(
                    f"On hyperparameter iteration {h} of {self.hparam_steps}...",
                    f"({h / self.hparam_steps:.1%})",
                )
                p = self.attack_alg.craft(x + p, y)
                success = self.hparam_criterion(x, p, b, y)
                hparam = self.attack.hparams[self.hparam]
                ub[success] = ub[success].minimum(hparam[success])
                lb[~success] = lb[~success].maximum(hparam[~success])
                hparam.copy_(ub.add(lb).div(2))
                print(
                    "Hyperparameter {self.hparam} updated "
                    f"({success.sum().div(success.numel()):.2%} success)"
                )

                # store the best adversarial perturbations seen thus far
                update = self.restart_criterion(x, p, b, y)
                b[update] = p[update]
                print(
                    f"Found {update.sum()} better adversarial examples!",
                    "(+{update.sum().div(update.numel()):.2%})",
                )
        return x + b.nan_to_num_(nan=None, posinf=0)

    def max_loss(self, x, p, b, y):
        """
        This method serves as a restart criterion as used in
        https://arxiv.org/pdf/1706.06083.pdf. Specifically, it returns the set
        of perturbations whose loss is maximal. Notably, if the loss is to be
        minimized and early termination is disabled, then the set of
        pertrubations whose loss is minimal is returned instead.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param p: candidate set of perturbation vectors
        :type p: torch Tensor object (n, m)
        :param b: best set of perturbation vectors
        :type b: torch Tensor object (n, m )
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        ploss, bloss = (self.loss(self.model(x + i), y) for i in (p, b))
        return ploss.gt(bloss) if self.loss.max_obj else ploss.lt(bloss)

    def min_norm(self, x, p, b, y):
        """
        This method serves as a restart criterion as used in
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
        :type y: Pytorch Tensor object (n,)
        :return: the perturbations whose loss is maximal
        :rtype: torch Tensor object (n,)
        """
        misclassified = self.model(x + p).argmax(1).ne(y)
        return misclassified.logical_and_(p.norm(self.norm, 1) < b.norm(self.norm, 1))

    def misclassified(self, x, p, y):
        """
        This method serves as a hyperparameter criterion as used in
        https://arxiv.org/pdf/1608.04644.pdf. Specifically, it returns the set
        of adversarial examples that are misclassified.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param p: set of perturbation vectors
        :type p: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
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
    :func:`linf`: clamps inputs to domain and linf threat model
    :func:`l0`: clamps inputs to domain and l0 threat model
    :func:`l2`: clamps inputs to domain and l2 threat model
    :func:`progress`: record various statistics on crafting progress
    :func:`tanh_p`: extracts perturbation from input when using CoV
    """

    def __init__(
        self,
        alpha,
        change_of_variables,
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
        :param verbosity: print attack statistics every verbosity%
        :type verbosity: float

        These subsequent parameters define the components of a Traveler object:

        :param alpha: learning rate of the optimizer
        :type alpha: float
        :param change_of_variables: whether to map inputs to tanh-space
        :type change_of_variables: bool
        :param optimizer_alg: optimization algorithm to use
        :type optimizer_alg: optimizer module class
        :param random_start: whether to randomly perturb initial inputs
        :type random_start: bool

        Finally, the following parameters define Surface objects:

        :param loss_func: objective function to differentiate
        :type loss_func: loss module class
        :param norm: lp-space to project gradients into
        :type norm: surface module callable
        :param model: neural network
        :type model: dlm LinearClassifier-inherited object
        :param saliency_map: desired saliency map heuristic
        :type saliency_map: saliency module class

        To easily identify attacks, __repr__ is overriden and instead returns
        an abbreviation computed by concatenating the first letter (or two, if
        there is a name collision) of the following parameters, in order: (1)
        optimizer, (2) if random start was used, (3) if change of variables
        was applied, (4) loss function, (5) saliency map, and (6) norm used.
        Combinations that yield known attacks are labeled as such (see __repr__
        for more details).

        :return: an attack
        :rtype: Attack object
        """

        # set & save attack parameters, and build short attack name
        self.batch_size = batch_size
        self.change_of_variables = change_of_variables
        self.epochs = epochs
        self.epsilon = epsilon
        self.et = early_termination
        self.lp = {surface.l0: 0, surface.l2: 2, surface.linf: torch.inf}[norm]
        self.project = getattr(self, norm.__name__)
        self.results = {m: [] for m in ("acc", "aloss", "mloss", "l0", "l2", "linf")}
        self.verbosity = int(epochs * verbosity)
        self.clip = torch.tensor(
            (-torch.inf, torch.inf) if change_of_variables else (0, 1)
        ).unbind()
        self.components = {
            "change of variables": change_of_variables,
            "loss function": loss_func.__name__,
            "optimizer": optimizer_alg.__name__,
            "random start": random_start,
            "saliency map": saliency_map.__name__,
            "target norm": "l∞" if norm == surface.linf else norm.__name__,
        }
        name = (
            self.components["optimizer"][0],
            "R" if self.components["random start"] else "r̶",
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
        self.params = {"α": alpha, "ε": epsilon, "epochs": epochs, "min-dist": self.et}

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
            "norm": self.lp,
            "saliency_map": saliency_map,
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
            change_of_variables, optimizer_alg, random_start * epsilon
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
        initilizations (i.e., change of variables and random start), and
        iterates an epoch number of times.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: Pytorch Tensor object (n,)
        :return: set of perturbation vectors
        :rtype: torch Tensor object (n, m)
        """

        # initialize perturbations, ranges, inputs, and batches
        x = x.detach().clone()
        p = torch.zeros_like(x)
        mins, maxs = (x.min(0).values, x.max(0).values)
        x.sub_(mins).div_(maxs.sub(mins))
        batch_size = x.size(0) if self.batch_size == -1 else self.batch_size
        num_batches = -(-x.size(0) // batch_size)
        o = torch.where(
            self.surface.model(x).argmax(1).ne(y).unsqueeze(1),
            torch.zeros_like(x),
            torch.full_like(x, torch.inf),
        )

        # configure batches, attach objects, & apply perturbation inits
        print(f"Crafting {x.size(0)} adversarial examples with {self}...")
        xi, yi, pi, oi = (t.split(batch_size) for t in (x, y, p, o))
        for b, (xb, yb, pb, ob) in enumerate(zip(xi, yi, pi, oi)):
            print(f"On batch {b + 1} of {num_batches} {(b + 1) / num_batches:.1%}...")
            cnb, cxb = (c.sub(xb) for c in self.clip)
            self.surface.initialize((mins, maxs), (cnb, cxb), pb)
            self.traveler.initialize(xb, pb)
            pb.clamp_(cnb, cxb)
            self.project(xb, pb)
            self.progress(b, 0, xb, yb, pb, ob)

            # compute peturbation updates and record progress
            for e in range(1, self.epochs + 1):
                self.surface(xb, yb, pb)
                self.traveler()
                pb.clamp_(cnb, cxb)
                self.project(xb, pb)
                progress = self.progress(b, e, xb, yb, pb, ob)
                None if not self.verbosity else print(
                    f"Epoch {e:{len(str(self.epochs))}} / {self.epochs} {progress}"
                ) if not e % self.verbosity else print(
                    f"Epoch {e}... ({e / self.epochs:.1%})", end="\r"
                )

        # compute final statistics, set failed perturbations to zero, & return
        for m in self.results:
            d = y.numel() if m in ("acc", "l0", "l2", "linf") else 1
            self.results[m] = [sum(s) / d for s in zip(*self.results[m])]
        return o.nan_to_num_(nan=None, posinf=0)

    def linf(self, x, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l∞ threat model (i.e., epsilon). Specifically,
        perturbation vectors whose l∞-norms exceed the threat model are
        projected back onto the l∞-ball. This is done by clipping perturbation
        vectors by ±epsilon. Notably, when using change of variables, we map
        clipped perturbation vectors into the tanh space and set the vectors to
        the computed result. In other words, after p is clipped according to
        ±epsilon, it is then set to:

                    p = ArcTanh((p - x) * 2 - 1) - (Tanh(x) + 1) / 2

        :param x: adversarial examples in tanh-space
        :type x: torch Tensor object (n, m)
        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """
        p_real = self.tanh_p(x, p) if self.change_of_variables else p
        p_real.clamp_(-self.epsilon, self.epsilon)
        p.copy_(self.tanh_p(x, p, True)) if self.change_of_variables else None
        return None

    def l0(self, x, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l0 threat model (i.e., epsilon). Specifically, this
        method sets any newly perturbed component of perturbation vectors to
        zero if such a perturbation exceeds the specified l0-threat model. To
        know which components are "newly" perturbed, we save a reference to the
        perturbation vector computed at the iteration prior.

        :param x: adversarial examples in tanh-space (unused)
        :type x: torch Tensor object (n, m)
        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """
        i = p.norm(0, 1) > self.epsilon
        p[i] = self.prev_p[i].where(self.prev_p[i] == 0, p[i]) if i.any() else 0.0
        self.prev_p = p.detach().clone()
        return None

    def l2(self, x, p):
        """
        This method projects perturbation vectors so that they are complaint
        with the specified l2 threat model (i.e., epsilon). Specifically,
        perturbation vectors whose l2-norms exceed the threat model are
        projected back onto the l2-ball. This is done by scaling such
        perturbation vectors by their l2-norms times epsilon. Notably, when
        using change of variables, we map scaled perturbation vectors into the
        tanh space and set the vectors to the computed result. In other words,
        after p is scaled according to epsilon, it is then set to:

                    p = ArcTanh((x + p) * 2 - 1) - (Tanh(x) + 1) / 2

        :param x: adversarial examples in tanh-space
        :type x: torch Tensor object (n, m)
        :param p: perturbation vectors
        :type p: torch Tensor object (n, m)
        :return: None
        :rtype: NoneType
        """
        p_real = self.tanh_p(x, p) if self.change_of_variables else p
        norm = p_real.norm(2, 1, True)
        p.copy_(p.where(norm <= self.epsilon, p_real.div(norm).mul(self.epsilon)))
        if self.change_of_variables:
            p.copy_(p.where(norm <= self.epsilon, self.tanh_p(x, p, True)))
        return None

    def progress(self, batch, epoch, xb, yb, pb, ob):
        """
        This method records various statistics on the adversarial example
        crafting process, updates final perturbations when using early
        termination, and returns a formatted string concisely representing the
        state of the attack. Specifically, this computes the following at each
        epoch (and the change since the last epoch): (1) model accuracy, (2)
        model loss, (3) attack loss, (3) l0, l2, and l∞ norms. Moreover, it
        also updates the early termination state of the current batch since
        model accuracy is computed.

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

        # compute absolute stats (averaged on exit) & update early termination
        logits = self.surface.model(self.surface.transform(xb + pb))
        mloss = self.surface.model.loss(logits, yb).item()
        ali = self.surface.loss(logits, yb)
        aloss = ali.sum().item()
        correct = logits.argmax(1).eq(yb)
        acc = correct.sum().item()

        # extract perturbation vectors from tanh-space if using cov
        pb = self.tanh_p(xb, pb) if self.change_of_variables else pb
        nb = [pb.norm(n, 1).sum().item() for n in (0, 2, torch.inf)]

        # update early termination and output buffer
        if self.et:
            smaller = pb.norm(self.lp, 1).lt(ob.norm(self.lp, 1))
            update = (~correct).logical_and_(smaller)
        else:

            # only map x out of tanh-space since perturbations already are
            ologits = self.surface.model(
                self.surface.min_max_scale(
                    self.surface.cov(xb) + ob.nan_to_num(posinf=0)
                )
            )
            oloss = self.surface.loss(ologits, yb)
            update = oloss.lt(ali) if self.surface.loss.max_obj else oloss.gt(ali)
        ob[update] = pb[update]

        # extend result lists; first by batch, then by epoch
        for m, s in zip(self.results.keys(), (acc, mloss, aloss, *nb)):
            try:
                self.results[m][batch].append(s)
            except IndexError:
                self.results[m].append([s])

        # build str representation and return
        return (
            f"Accuracy: {acc / yb.numel():.1%} "
            f"({(acc - sum(self.results['acc'][batch][-2:-1])) / yb.numel():+.1%}) "
            f"Model Loss: {mloss:.2f} "
            f"({mloss - sum(self.results['mloss'][batch][-2:-1]):+6.2f}) "
            f"{self.name} Loss: {aloss:.2f} "
            f"({aloss - sum(self.results['aloss'][batch][-2:-1]):+6.2f}) "
            f"l0: {nb[0] / yb.numel():.2f} "
            f"({(nb[0] - sum(self.results['l0'][batch][-2:-1])) / yb.numel():+.2f}) "
            f"l2: {nb[1] / yb.numel():.2f} "
            f"({(nb[1] - sum(self.results['l2'][batch][-2:-1])) / yb.numel():+.2f}) "
            f"l∞: {nb[2] / yb.numel():.2f} "
            f"({(nb[2] - sum(self.results['linf'][batch][-2:-1])) / yb.numel():+.2f})"
        )

    def tanh_p(self, x, p, into=False):
        """
        This method extracts the perturbation vector when using change of
        variables. When using the technique, the computed perturbation is in
        the tanh-space (which is convenient when returning adversarial examples
        directly). To properly compute progress statistics as well as
        projecting onto l2-norm balls (parameterized by epsilon), this method
        extracts the perturbation vector in the real space by first mapping the
        current input out of the tanh-space and subtracting it from the
        original input. Moreover, to facilitate mapping perturbations into the
        tanh-space, this also supports an out-of-place-non-scaled version of
        the procedure defined in tanh_map.

        :param x: initial inputs in tanh-space
        :type x: torch Tensor object (n, m)
        :param p: perturbations in tanh-space
        :type p: torch Tensor object (n, m)
        :param into: whether to map into (or out of) the tanh-space
        :type into: bool
        :return: perturbation vector mapped out of tanh-space
        :rtype: torch Tensor object (n, m)
        """
        return (
            traveler.tanh_space(x).add(p).mul(2).sub(1).arctanh().sub(x)
            if into
            else traveler.tanh_space(x.add(p)).sub(traveler.tanh_space(x))
        )


def attack_builder(
    alpha=None,
    epochs=None,
    early_termination=None,
    epsilon=None,
    model=None,
    change_of_variables_enabled=(True, False),
    optimizers=(
        optimizer.Adam,
        optimizer.BackwardSGD,
        optimizer.MomentumBestStart,
        optimizer.SGD,
    ),
    random_start_enabled=(True, False),
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
    techniques used within the AML community, such as random start and change
    of variables. The combinations of supported components are shown below:

        Traveler Components:
        Change of variables: Enabled or disabled
        Optimizer: Adam, Backward Stochastic Gradient Descent,
                    Momentum Best Start, and Stochastic Gradient Descent
        Random start: Enabled or disabled

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
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :type model: dlm LinearClassifier-inherited object
    :return: a generator yielding attack combinations
    :rtype: generator of Attack objects
    """

    # generate combinations of components and instantiate Attack objects
    num_attacks = (
        len(optimizers)
        * len(random_start_enabled)
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
        random_start,
        saliency_map,
    ) in itertools.product(
        change_of_variables_enabled,
        losses,
        norms,
        optimizers,
        random_start_enabled,
        saliency_maps,
    ):
        yield Attack(
            alpha=alpha,
            change_of_variables=change_of_variables,
            early_termination=early_termination,
            epochs=epochs,
            loss=loss_func,
            model=model,
            norm=norm,
            optimizer=optimizer_alg,
            random_start=random_start,
            saliency_map=saliency_map,
            verbosity=verbosity,
        )


def apgdce(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Auto-PGD with Cross-Entropy loss
    (APGD-CE), as shown in https://arxiv.org/abs/2003.01690. Specifically,
    APGD-CE: does not use change of variables, uses the Momentum Best Start
    optimizer, uses random start, uses Cross-Entropy loss, uses l∞ norm, and
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
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: APGD-CE attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        change_of_variables=False,
        optimizer_alg=optimizer.MomentumBestStart,
        random_start=True,
        loss_func=loss.CELoss,
        norm=surface.linf,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def apgddlr(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Auto-PGD with the Difference of
    Logits Ratio loss (APGD-DLR), as shown in https://arxiv.org/abs/2003.01690.
    Specifically, APGD-DLR: does not use change of variables, uses the Momentum
    Best Start optimizer, uses random start, uses Difference of Logits ratio
    loss, uses l∞ norm, and uses the Identity saliency map.

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
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: APGD-DLR attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.DLRLoss,
        model=model,
        norm=surface.linf,
        optimizer_alg=optimizer.MomentumBestStart,
        random_start=True,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def bim(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build the Basic Iterative Method (BIM),
    as shown in (https://arxiv.org/pdf/1611.01236.pdf) Specifically, BIM: does
    not use change of variables, uses the Stochastic Gradient Descent
    optimizer, does not use random start, uses Cross Entropyy loss, uses l∞
    norm, and uses the Identity saliency map.

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
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: BIM attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CELoss,
        model=model,
        norm=surface.linf,
        optimizer_alg=optimizer.SGD,
        random_start=False,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def cwl2(
    alpha=None,
    epochs=None,
    epsilon=None,
    model=None,
    hparam_bounds=(0, 1e10),
    hparam_steps=5,
    verbosity=1,
):
    """
    This function serves as an alias to build Carlini-Wagner l₂ (CW-L2), as
    shown in (https://arxiv.org/pdf/1608.04644.pdf) Specifically, CW-L2: uses
    change of variables, uses the Adam optimizer, does not use random start,
    uses Carlini-Wagner loss, uses l₂ norm, and uses the Identity saliency map.
    It also performs binary search over c, which controls the importance of
    inducing misclassification over minimizing norm.

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
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: Carlini-Wagner l₂ attack
    :rtype: Attack objects
    """
    return Adversary(
        hparam="c",
        hparam_bounds=hparam_bounds,
        hparam_criterion=Adversary.misclassified,
        hparam_steps=hparam_steps,
        num_restarts=0,
        restart_criterion=Adversary.min_norm,
        verbose=False,
        alpha=alpha,
        change_of_variables=True,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CWLoss,
        model=model,
        norm=surface.l2,
        optimizer_alg=optimizer.Adam,
        random_start=False,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def df(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build DeepFool (DF), as shown in
    (https://arxiv.org/pdf/1511.04599.pdf) Specifically, DF: does not use
    change of variables, uses the Stochastic Gradient Descent optimizer, does
    not use random start, uses Identity loss, uses l₂ norm, and uses the
    DeepFool saliency map.

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
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.l2,
        optimizer_alg=optimizer.SGD,
        random_start=False,
        saliency_map=saliency.DeepFoolSaliency,
        verbosity=verbosity,
    )


def fab(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Fast Adaptive Boundary (FAB), as
    shown in (https://arxiv.org/pdf/1907.02044.pdf) Specifically, FAB: does not
    use change of variables, uses the Backward Stochastic Gradient Descent
    optimizer, does not use random start, uses Identity loss, uses l₂ norm, and
    uses the DeepFool saliency map.

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
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        early_termination=True,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.l2,
        optimizer_alg=optimizer.BackwardSGD,
        random_start=False,
        saliency_map=saliency.DeepFoolSaliency,
        verbosity=verbosity,
    )


def pgd(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build Projected Gradient Descent (PGD),
    as shown in (https://arxiv.org/pdf/1706.06083.pdf) Specifically, PGD: does
    not use change of variables, uses the Stochastic Gradient Descent
    optimizer, uses random start, uses Identity loss, uses l∞ norm, and uses
    the Identity saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        change_of_variables=False,
        early_termination=False,
        epochs=epochs,
        epsilon=epsilon,
        loss_func=loss.CELoss,
        model=model,
        norm=surface.linf,
        optimizer_alg=optimizer.SGD,
        random_start=True,
        saliency_map=saliency.IdentitySaliency,
        verbosity=verbosity,
    )


def jsma(alpha=None, epochs=None, epsilon=None, model=None, verbosity=1):
    """
    This function serves as an alias to build the Jacobian-based Saliency Map
    Approach (JSMA), as shown in (https://arxiv.org/pdf/1511.07528.pdf)
    Specifically, the JSMA: does not use change of variables, uses the
    Stochastic Gradient Descent optimizer, does not use random start, uses
    Identity loss, uses l0 norm, and uses the Jacobian saliency map.

    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack objects
    """
    return Attack(
        alpha=alpha,
        early_termination=True,
        epochs=epochs,
        change_of_variables=False,
        epsilon=epsilon,
        loss_func=loss.IdentityLoss,
        model=model,
        norm=surface.l0,
        optimizer_alg=optimizer.SGD,
        random_start=False,
        saliency_map=saliency.JacobianSaliency,
        verbosity=verbosity,
    )
