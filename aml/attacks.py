"""
This module is the core of the adversarial machine learning repo. It defines
the attacks proposed in https://arxiv.org/pdf/2209.04521.pdf.
"""

import itertools

import pandas
import torch

import aml.loss as loss
import aml.optimizer as optimizer
import aml.surface as surface
import aml.traveler as traveler


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
    :func:`__getattr__`: return Attack object attributes
    :func:`__repr__`: returns the threat model
    :func:`__setattr__`: sets attributes of Attack objects
    :func:`binary_search`: optimizes hyperparamters via binary search
    :func:`craft`: returns a set of adversarial examples
    :func:`max_loss`: restart rule used in PGD attack
    :func:`min_norm`: restart rule used in FAB attack
    :func:`misclassified`: hyperparameter rule used in CW-L2 attack
    :func:`progress`: updates output dataframe
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

        :param bestupdate: update rule for the best adversarial examples
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
        hparam_steps = 0 if hparam_steps is None else hparam_steps

        # set attributes and update total number of iterations
        self.__dict__["atk"] = attack
        self.__dict__["craft_p"] = (
            (lambda x, y, b, reset: (attack.craft(x, y, b, reset),))
            if hparam is None
            else self.binary_search
        )
        self.__dict__["best_update"] = best_update
        self.__dict__["hparam"] = hparam
        self.__dict__["hparam_bounds"] = hparam_bounds
        self.__dict__["hparam_steps"] = hparam_steps + 1
        self.__dict__["hparam_update"] = hparam_update
        self.__dict__["num_restarts"] = num_restarts + 1
        self.__dict__["verbose"] = verbose
        self.__dict__["params"] = {
            "best_update": best_update.__name__,
            "hparam": hparam,
            "hparam_bounds": hparam_bounds,
            "hparam_update": hparam_update.__name__,
            "hparam_steps": hparam_steps,
            "num_restarts": num_restarts,
            "attack": repr(attack),
        }
        self.__dict__["atk"].total *= (
            self.__dict__["num_restarts"] * self.__dict__["hparam_steps"]
        )
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
        return self.__getattribute__("atk").__getattribute__(name)

    def __repr__(self):
        """
        This method returns a concise string representation of adversary
        and attack parameters.

        :return: adversarial and attack parameters
        :rtype: str
        """
        return f"Adversary({', '.join(f'{p}={v}' for p, v in self.params.items())})"

    def __setattr__(self, name, value):
        """
        This method overrides setting Adversary attributes so that they are
        changed in attack objects instead. As Adversary objects ostenisbly
        serve as intelligent looping objects for Attack objects, any attributes
        that may be set (or changed) at runtime should be passed to Attack
        objects instead (as Adversary objects have no parameters that could be
        meaningfully adjusted at runtime).

        :param name: name of attribute to update
        :type name: str
        :param value: value to update attribute
        :type value: misc
        :return: None or value
        :rtype: NoneType or type(value)
        """
        setattr(self.atk, name, value)

    def binary_search(self, x, y, b, reset):
        """
        This method performs binary search on a hyperparameter, such as c used
        in Carlini-Wagner loss (https://arxiv.org/pdf/1608.04644.pdf). By
        convention, success (encoded as True by any update rule) decreases the
        hyperparameter, while failure (encoded as False) will increase it.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param b: best set of perturbation vectors
        :type b: torch Tensor object (n, m )
        :param reset: whether to reset the internal attack state
        :type reset: bool
        :return: adversarial perturbations
        :rtype: generator of torch Tensor objects (n, m)
        """
        lb, ub = (
            torch.tensor(self.hparam_bounds, device=b.device)
            .repeat(y.numel(), 1)
            .unbind(1)
        )
        for h in range(1, self.hparam_steps + 1):
            p = self.atk.craft(x, y, b, reset)
            self.atk.hparam = f"BS Step {h} "
            self.verbose and print(
                f"On hyperparameter iteration {h} of {self.hparam_steps}...",
                f"({h / self.hparam_steps:.1%})",
            )
            success = self.hparam_update(x, p, y)
            hparam = self.atk.hparams[self.hparam]
            ub[success] = ub[success].minimum(hparam[success])
            lb[~success] = lb[~success].maximum(hparam[~success])
            hparam.copy_(ub.add(lb).div(2))
            self.verbose and print(
                f"Hyperparameter {self.hparam} updated.",
                f"{success.mean(dtype=torch.float):.2%} success",
            )
            yield p

    def craft(self, x, y, reset=False):
        """
        This method represents the core of the Adversary class. Specifically,
        this class produces the most effective adversarial examples,
        parameterized by restarts and hyperparameter optimization.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param reset: whether to reset the internal attack state
        :type reset: bool
        :return: adversarial perturbations
        :rtype: torch Tensor object (n, m)
        """

        # instantiate bookkeeping and iterate over restarts & hyperparameters
        x = x.clone()
        b = torch.zeros_like(x)
        b[self.model.accuracy(x, y, as_tensor=True)] = torch.inf
        self.atk.step = 0
        self.last_res = None
        for r in range(self.num_restarts):
            self.atk.reset()
            self.atk.restart = f"Restart {r} " if r > 0 else ""
            r > 0 and self.verbose and print(
                f"On restart iteration {r} of {self.num_restarts - 1}...",
                f"({r / (self.num_restarts - 1):.1%})",
            )

            # store the best adversarial perturbations seen thus far
            for p in self.craft_p(x, y, b, reset):
                self.atk.step += 1
                update = self.best_update(x, p, b, y)
                non_adv = b.isinf().any(1)
                b[update] = p[update]
                new = update.logical_and(non_adv).mean(dtype=torch.float)
                improved = update.logical_and(~non_adv).mean(dtype=torch.float)
                self.verbose and print(
                    f"Found {update.sum()} better adversarial examples!",
                    f"(+{update.mean(dtype=torch.float):.2%})",
                    f"(New {new:.2%}, Improved {improved:.2%})",
                )

                # override output dataframe for meaningful results
                self.statistics and self.progress()
        return b.nan_to_num_(nan=None, posinf=0)

    def max_loss(self, x, p, b, y):
        """
        This method serves as a restart rule as used in
        https://arxiv.org/pdf/1706.06083.pdf. Specifically, it returns the set
        of misclassified perturbations whose loss has improved.

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
        ploss = self.atk.surface.loss(self.atk.model(x + p), y)
        bloss = self.atk.surface.loss(self.atk.model(x + b.nan_to_num(posinf=0)), y)
        return self.misclassified(x, p, y).logical_and_(
            ploss.gt(bloss) if self.atk.surface.loss.max_obj else ploss.lt(bloss)
        )

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
        return ~self.model.accuracy(x + p, y, as_tensor=True)

    def progress(self):
        """
        This method updates the output results dataframe so that the recorded
        results can be meaningfully interpreted. Specifically, since the output
        dataframe is computed from the output buffer, output results dataframes
        are not particularly meaningful. For attacks that use random restarts
        or hyperparamter optimization, attack performance can be meaningfuly
        interpreted by taking the best results across batch results dataframes
        instead. To this end, this method records the best accuracy, model loss
        attack loss, and norms observed across batch results dataframes.

        :param results: dataframe of best batch performance
        :type results: pandas DataFrame object
        :return: None
        :rtype: NoneType
        """
        try:
            last_out = self.last_res.iloc[-1]
            self.last_res = self.last_res.assign(
                **(self.last_res - (last_out - self.res)).drop(columns="epoch")
            )
            self.res = self.last_res
        except AttributeError:
            self.last_res = self.res.copy()
        return None


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
    :func:`reset`: reset attack state (akin to reinstantiation)
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
        alpha_override=True,
        statistics=False,
        verbosity=0.25,
    ):
        """
        This method instantiates an Attack object with a variety of parameters
        necessary for building and coupling Traveler and Surface objects. The
        following parameters define high-level bookkeeping parameters across
        attacks:

        :param alpha_override: override manual alpha for certain components
        :type alpha_override: bool
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

        # set alpha to 1 with DF smap with non-adaptive-LR optimizer (mostly)
        alpha = (
            1.0
            if alpha_override
            and saliency_map is surface.DeepFoolSaliency
            and optimizer_alg in {optimizer.BackwardSGD, optimizer.SGD}
            else alpha
        )

        # set & save attack parameters, and build short attack name
        self.alpha = alpha
        self.device = model.device
        self.epochs = epochs
        self.epsilon = epsilon
        self.et = early_termination
        self.hparam = ""
        self.lp = {surface.L0: 0, surface.L2: 2, surface.Linf: torch.inf}[norm]
        self.restart = ""
        self.step = 0
        self.statistics = statistics
        self.total = epochs
        self.verbosity = max(int(epochs * verbosity), 1 if verbosity else 0)
        self.components = {
            "loss function": loss_func.__name__,
            "optimizer": optimizer_alg.__name__,
            "random start": random_start.__name__,
            "saliency map": saliency_map.__name__,
            "target norm": "L∞" if norm == surface.Linf else norm.__name__,
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

        # save components and set parameters for __repr__
        self.loss_class = loss_func
        self.model = model
        self.norm_class = norm
        self.optimizer_class = optimizer_alg
        self.random_class = random_start
        self.saliency_class = saliency_map
        self.params = {
            "α": self.alpha,
            "ε": self.epsilon,
            "epochs": self.epochs,
        }
        self.reset()
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
        p = ", ".join(f"{p}={v}" for p, v in self.params.items())
        return f"{self.name}({p}, early termination={self.et}, device={self.device})"

    def craft(self, x, y, o=None, reset=False):
        """
        This method crafts adversarial examples, as defined by the attack
        parameters and the instantiated Traveler and Surface objects.
        Specifically, it creates a copy of x, optionally initializes the output
        buffer to a set of optimal perturbations (useful for attacks that
        perform shrinking start), performs surface & traveler initilizations,
        and finally iterates an epoch number of times.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param o: initialize output buffer to saved adversarial examples
        :type o: torch Tensor object (n, m)
        :param reset: whether to reset the internal attack state
        :type reset: bool
        :return: set of perturbation vectors
        :rtype: torch Tensor object (n, m)
        """

        # init perturbations, set misclassified outputs to 0, & compute ranges
        reset and self.reset()
        x = x.detach().clone()
        p = torch.zeros_like(x)
        correct = self.model.accuracy(x, y, as_tensor=True).unsqueeze_(1)
        o = torch.full_like(p, torch.inf).where(correct, p) if o is None else o.clone()
        mins, maxs = x.min(0).values.clamp(max=0), x.max(0).values.clamp(min=1)

        # configure batch & output results dataframes
        cols = "epoch", "accuracy", "model_loss", "attack_loss", "l0", "l2", "linf"
        self.b_res = pandas.DataFrame(0.0, index=range(self.epochs + 1), columns=cols)
        self.res = pandas.DataFrame(0.0, index=range(self.epochs + 1), columns=cols)

        # attach objects and  apply perturbation initializations
        self.verbosity and print(f"Crafting {len(x)} adversarial examples with {self}")
        cnb, cxb = mins.sub(x), maxs.sub(x)
        self.surface.initialize((cnb, cxb), p)
        self.traveler.initialize(p, o)
        p.clamp_(cnb, cxb)
        self.update(x, y, p, o)
        self.statistics and self.progress(0, x, y, p, o)

        # compute peturbation updates and record progress
        for e in range(1, self.epochs + 1):
            self.surface(x, y, p)
            self.traveler()
            p.clamp_(cnb, cxb)
            not self.et and self.project(p)
            self.update(x, y, p, o)
            prog = self.progress(e, x, y, p, o) if self.statistics else ""
            (
                print(
                    f"{self.name}: {self.restart}{self.hparam}"
                    f"Epoch {e:{len(str(self.epochs))}} / {self.epochs} {prog:<15}"
                )
                if self.verbosity != 0 and e % self.verbosity == 0
                else print(
                    f"{self.name}: {self.restart}{self.hparam}"
                    f"Epoch {e}... ({(e + self.epochs * self.step) / self.total:.1%})",
                    end="\x1b[K\r",
                )
            )

        # set failed perturbations to zero and return
        return o.nan_to_num_(nan=None, posinf=0)

    def progress(self, e, x, y, p, o):
        """
        This method measures various statistics on the adversarial crafting
        process & a formatted string concisely representing the state of the
        attack. Specifically, this computes the following at each epoch (and
        the change since the last epoch): (1) model accuracy, (2) model loss,
        (3) attack loss, (3) l0, l2, and l∞ norms. Notably, due to performance
        bugs in PyTorch (https://github.com/pytorch/pytorch/issues/51509) and
        the number of forward passes necessary to compute interesting
        statistics, this method can have a high performance impact.

        :param e: current epoch
        :type e: int
        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param p: current batch of perturbations
        :type p: torch Tensor object (n, m)
        :param o: current output perturbations
        :type o: torch Tensor object (n, m)
        :return: print-ready statistics
        :rtype: str
        """

        # project batch onto lp-threat model, compute stats, & update results
        norms = (0, 2, torch.inf)
        p = self.project(p.clone())
        logits = self.surface.model(x + p)
        mloss = self.surface.model.loss(logits, y).item()
        aloss = self.surface.loss(logits, y).sum().item()
        pacc = logits.argmax(1).eq_(y).mean(dtype=torch.float).item()
        pn = [p.norm(n, 1).mean().item() for n in norms]
        self.b_res.iloc[e] = e, pacc, mloss, aloss, *pn

        # compute output perturbation stats and update results
        ologits = self.surface.model(x + o.nan_to_num(posinf=0))
        omloss = self.surface.model.loss(ologits, y).item()
        oaloss = self.surface.loss(ologits, y).sum().item()
        oacc = ologits.argmax(1).eq_(y).mean(dtype=torch.float).item()
        on = [o.nan_to_num(posinf=0).norm(n, 1).mean().item() for n in norms]
        self.res.iloc[e] = e, oacc, omloss, oaloss, *on

        # build str representation and return
        return (
            f"Output Acc: {oacc:.1%} Batch Acc: {pacc:.1%} "
            f"({(pacc - self.b_res.accuracy.iloc[e - 1]):+.1%}) "
            f"Model Loss: {mloss:.2f} "
            f"({mloss - self.b_res.model_loss.iloc[e - 1]:+6.2f}) "
            f"{self.name} Loss: {aloss:.2f} "
            f"({aloss - self.b_res.attack_loss.iloc[e - 1]:+6.2f}) "
            f"l0: {pn[0]:.2f} ({pn[0] - self.b_res.l0.iloc[e - 1]:+.2f}) "
            f"l2: {pn[1]:.2f} ({pn[1] - self.b_res.l2.iloc[e - 1]:+.2f}) "
            f"l∞: {pn[2]:.2f} ({pn[2] - self.b_res.linf.iloc[e - 1]:+.2f})"
        )

    def reset(self):
        """
        This method resets the state of all objects inside attacks.
        Conceptually, this is akin to instantiating a new attack object.
        Specifically, this initializes: (1) attack loss, (2) lp-norm filter,
        (3) random start strategy, (4) saliency map, (5) optimizer, (6)
        traveler, (7) surface, (8) projection method (attached to lp-norm
        filter), (9) any registered hyperparameters, and (10) attack parameters
        referenced by __repr__.

        :return: None
        :rtype: NoneType
        """

        # instantiate traveler, surface, and necessary subcomponents
        attack_loss = self.loss_class(classes=self.model.classes, device=self.device)
        norm = self.norm_class(
            device=self.device, epsilon=self.epsilon, maximize=self.loss_class.max_obj
        )
        random_start = self.random_class(
            device=self.device, epsilon=self.epsilon, norm=self.lp
        )
        saliency_map = self.saliency_class(classes=self.model.classes, p=self.lp)
        aml_opt = {
            "attack_loss": attack_loss,
            "epochs": self.epochs,
            "epsilon": 1.0 if self.lp == 0 else self.epsilon,
            "model": self.model,
            "norm": self.lp,
            "smap": saliency_map,
        }
        optimizer_params = {
            "lr": self.alpha,
            "maximize": attack_loss.max_obj,
            "params": (torch.zeros(1),),
        } | (aml_opt if self.optimizer_class.__module__ == optimizer.__name__ else {})
        optimizer_alg = self.optimizer_class(**optimizer_params)
        self.traveler = traveler.Traveler(optimizer_alg, random_start)
        self.surface = surface.Surface(attack_loss, self.model, norm, saliency_map)

        # set projection method, collect hyperparameters, and set parameters
        self.project = norm.project
        self.hparams = self.traveler.hparams | self.surface.hparams
        self.params = {
            "α": self.alpha,
            "ε": self.epsilon,
            "epochs": self.epochs,
        }
        return None

    def update(self, x, y, p, o):
        """
        This method updates the final perturbations returned by attacks.
        Specifically, the update is determined as: perturbations that are
        misclassified and (1) a smaller lp-norm than the current best
        perturbations for minimum-norm adversaries, or (2) the attack loss has
        improved for maximuim-loss adversaries.

        :param x: inputs to produce adversarial examples from
        :type x: torch Tensor object (n, m)
        :param y: the labels (or initial predictions) of x
        :type y: torch Tensor object (n,)
        :param p: current batch of perturbations
        :type p: torch Tensor object (n, m)
        :param o: current output perturbations
        :type o: torch Tensor object (n, m)
        loss than the current best perturbations are saved.
        :return: None
        :rtype: NoneType
        """
        p = self.project(p.clone())
        logits = self.surface.model(x + p)
        if self.et:
            update = p.norm(self.lp, dim=1).lt(o.norm(self.lp, dim=1))
        else:
            ploss = self.surface.loss(logits, y)
            ologits = self.surface.model(x + o.nan_to_num(posinf=0))
            oloss = self.surface.loss(ologits, y)
            update = ploss.gt(oloss) if self.surface.loss.max_obj else ploss.lt(oloss)
        update.logical_and_(logits.argmax(1).ne(y))
        o[update] = p[update]
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
    alpha_override=True,
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

    :param alpha_override: override manual alpha for certain components
    :type alpha_override: bool
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
    verbosity and print(f"Yielding {num_attacks} attacks...")
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
            alpha_override=alpha_override,
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


def get_attacks(
    alpha,
    attacks,
    early_termination,
    epochs,
    epsilon,
    model,
    alpha_override=True,
    statistics=False,
    verbosity=1,
):
    """
    This function serves as a wrapper to the attack_builder function via a
    string-based interface. Specifically, this function takes a string of
    attack names and returns a generator of Attack objects. The naming
    convention for attacks is as follows: {first letter of optimizer}-{first
    letter of random start strategy}-{first two letters of loss function}-
    {first letter of saliency map}-{second letter of norm}. Notably, wildcard
    characters (i.e., *) are supported for each component. For example, the
    string "M-*-CE-*-*" would return all attacks that use the Momentum
    optimizer and the Cross-Entropy loss function.

    :param alpha_override: override manual alpha for certain components
    :type alpha_override: bool
    :param alpha: learning rate of the optimizer
    :type alpha: float
    :param attacks: attack names
    :type attacks: str
    :param early_termination: whether misclassified inputs are perturbed
    :type early_termination: bool
    :param epochs: number of optimization steps to perform
    :type epochs: int
    :param epsilon: lp-norm ball threat model(s)
    :type epsilon: float or tuple of floats
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: a generator yielding attack combinations
    :rtype: generator of Attack objects
    """
    opt, random_start, loss_func, saliency_map, norm = attacks.split("-")
    optimizers = {
        o.__name__[0]: o
        for o in (
            optimizer.Adam,
            optimizer.BackwardSGD,
            optimizer.MomentumBestStart,
            optimizer.SGD,
        )
    }
    random_starts = {
        r.__name__[0]: r
        for r in (
            traveler.IdentityStart,
            traveler.MaxStart,
            traveler.ShrinkingStart,
        )
    }
    losses = {
        lo.__name__[:2]: lo
        for lo in (
            loss.CWLoss,
            loss.CELoss,
            loss.DLRLoss,
            loss.IdentityLoss,
        )
    }
    saliency_maps = {
        s.__name__[0]: s
        for s in (
            surface.DeepFoolSaliency,
            surface.IdentitySaliency,
            surface.JacobianSaliency,
        )
    }
    norms = {
        n.__name__[1]: n
        for n in (
            surface.L0,
            surface.L2,
            surface.Linf,
        )
    }
    params = dict(
        optimizers=optimizers.values() if opt == "*" else [optimizers[opt]],
        random_starts=(
            random_starts.values()
            if random_start == "*"
            else [random_starts[random_start]]
        ),
        losses=losses.values() if loss_func == "*" else [losses[loss_func]],
        saliency_maps=(
            saliency_maps.values()
            if saliency_map == "*"
            else [saliency_maps[saliency_map]]
        ),
        norms=norms.values() if norm == "*" else [norms[norm]],
    )
    return attack_builder(
        alpha_override=alpha_override,
        alpha=alpha,
        early_termination=early_termination,
        epochs=epochs,
        epsilon=epsilon,
        model=model,
        statistics=statistics,
        verbosity=verbosity,
        **params,
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
    :param epsilon: l∞ threat model
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
    :rtype: Adversary object
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
    :param epsilon: l∞ threat model
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
    :rtype: Adversary object
    """
    return Adversary(
        best_update=Adversary.max_loss,
        hparam=None,
        hparam_bounds=None,
        hparam_update=None,
        hparam_steps=None,
        num_restarts=num_restarts,
        verbose=True if verbosity else False,
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
    :param epsilon: l∞ threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: BIM attack
    :rtype: Attack object
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
    :param epsilon: l2 threat model
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
    :rtype: Attack object
    """
    return Adversary(
        best_update=Adversary.min_norm,
        hparam="c",
        hparam_bounds=hparam_bounds,
        hparam_steps=hparam_steps,
        hparam_update=Adversary.misclassified,
        num_restarts=0,
        verbose=True if verbosity else False,
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


def df(alpha, epochs, epsilon, model, statistics=False, verbosity=1):
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
    :param epsilon: l2 threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: DeepFool attack
    :rtype: Attack object
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
    :param epsilon: l2 threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param num_restarts: number of restarts to perform
    :type num_restarts: int
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: FAB attack
    :rtype: Adversary object
    """
    return Adversary(
        best_update=Adversary.min_norm,
        hparam=None,
        hparam_bounds=None,
        hparam_update=None,
        hparam_steps=None,
        num_restarts=num_restarts,
        verbose=True if verbosity else False,
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
    :param epsilon: l0 threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: JSMA attack
    :rtype: Attack object
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
    :param epsilon: l∞ threat model
    :type epsilon: float
    :param model: neural network
    :type model: dlm LinearClassifier-inherited object
    :param statistics: save attack progress (heavily increases compute)
    :type statistics: bool
    :param verbosity: print attack statistics every verbosity%
    :type verbosity: float
    :return: PGD attack
    :rtype: Attack object
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
