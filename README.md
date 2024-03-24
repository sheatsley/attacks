# Adversarial Machine Learning

_Adversarial Machine Learning_ (`aml`) is a repo for measuring the robustness
of deep learning models against white-box evasion attacks. Designed for
academics, it is principally designed for use in fundamental research to
understand _adversarial examples_, [inputs designed to cause models to make a
mistake](https://arxiv.org/abs/1412.6572). At its core, `aml` is based on a
series of techniques used in eight popular attacks:

1. [APGD-CE](https://arxiv.org/pdf/2003.01690.pdf) (Auto-PGD with CE loss)
2. [APGD-DLR](https://arxiv.org/pdf/2003.01690.pdf) (Auto-PGD with DLR loss)
3. [BIM](https://arxiv.org/pdf/1611.01236.pdf) (Basic Iterative Method)
4. [CW-L2](https://arxiv.org/pdf/1608.04644.pdf) (Carlini-Wagner with l2 norm)
5. [DF](https://arxiv.org/pdf/1511.04599.pdf) (DeepFool)
6. [FAB](https://arxiv.org/pdf/1907.02044.pdf) (Fast Adaptive Boundary)
7. [JSMA](https://arxiv.org/pdf/1511.07528.pdf) (Jacobian Saliency Map Approach)
8. [PGD](https://arxiv.org/pdf/1706.06083.pdf) (Projected Gradient Descent)

Notably, we have taken certain liberties in modifying the implementation of
these techniques to either improve their performance or their clarity (without
any cost in performance). Not only are these modifications designed to help
academics understand _why_ attacks perform the way that they do, but also to
[serve as abstractions for effortlessly building a vast space of new
attacks](https://arxiv.org/abs/2209.04521). At this time, the techniques based
on the eight attacks above enable construction of 432 total attacks (all of
which often decrease model accuracy to less than 1% with ~100 iterations at a
budget of 15% across the datasets found in [this
repo](https://github.com/sheatsley/datasets)). All of the information you need
to start using this repo is contained within this one ReadMe, ordered by
complexity (No need to parse through any ReadTheDocs documentation).

#### But wait, didn't [The Space of Adversarial Strategies](https://arxiv.org/abs/2209.04521) have 576 attacks?

Yes, our original paper did investigate 576 attacks. This repo has removed
support for [change of variables](https://arxiv.org/pdf/1608.04644.pdf) and
added support for (what we call) [shrinking
start](https://arxiv.org/pdf/1907.02044.pdf) random start strategy.
Empirically, we did not find _change of variables_ to offer any improvements
(with PyTorch) and its integration into the abstractions provided here was
rather complicated (without it, the repo is _substantially_ simpler).
_Shrinking start_ support was added with the new `Adversary` abstraction to
support attacks with: (1) non-deterministic components that perform multiple
starts (e.g., [FAB](https://arxiv.org/pdf/1907.02044.pdf)), and (2)
hyperparameters optimization (e.g.,
[CW-L2](https://arxiv.org/pdf/1608.04644.pdf)).

## Table of Contents

* [Quick Start](#quick-start)
* [Advanced Usage](#advanced-usage)
* [Repo Overview](#repo-overview)
* [Parameters](#parameters)
* [Misc](#misc)
* [Citation](#citation)

## Quick Start

This repo is, by design, to be interoperable with the following
[datasets](https://github.com/sheatsley/datasets) and
[models](https://github.com/sheatsley/models) repos (which are all based on
[PyTorch](https://github.com/pytorch/pytorch)). With some effort, you can
probably bring your own data and models, but it is not recommended if you're
just looking to start using this repo as easily as possible. Preferably,
install an editable version of this repo via `pip install -e`. Afterwards, you
can craft adversarial examples using any of the eight attacks above as follows:

    import aml
    import dlm
    import mlds
    import torch

    # load data
    mnist = mlds.mnist
    x_train = torch.from_numpy(mnist.train.data)
    y_train = torch.from_numpy(mnist.train.labels).long()
    x_test = torch.from_numpy(mnist.test.data)
    y_test = torch.from_numpy(mnist.test.labels).long()

    # instantiate and train a model
    hyperparameters = dlm.templates.mnist.cnn
    model = dlm.CNNClassifier(**hyperparameters)
    model.fit(x_train, y_train)

    # set attack parameters and produce adversarial perturbations
    step_size = 0.01
    number_of_steps = 30
    budget = 0.15
    pgd = aml.pgd(step_size, number_of_steps, budget, model)
    perturbations = pgd.craft(x_test, y_test)

    # compute some interesting statistics and publish a paper
    accuracy = model.accuracy(x_test + perturbations, y_test)
    mean_budget = perturbations.norm(torch.inf, 1).mean()

Other uses can be found in the
[examples](https://github.com/sheatsley/attacks/tree/main/examples) directory.

## Advanced Usage

Below are descriptions of some of the more subtle controls within this repo and
complex use cases.

* Early termination: when an instantiating an `Attack`, the `early_termination`
    flag determines whether attacks attempt to _minimize model accuracy_ or
    _maximize (model) loss_ (Also described as [maximum-confidence vs.
    minimum-distance](https://arxiv.org/pdf/1809.02861.pdf)). Specifically,
    attacks that "terminate early" return the set of misclassified inputs with
    the smallest norm (For example, perturbations for inputs that are initially
    misclassified are 0). Attacks in this regime include
    [CW-L2](https://arxiv.org/pdf/1608.04644.pdf),
    [DF](https://arxiv.org/pdf/1511.04599.pdf),
    [FAB](https://arxiv.org/pdf/1907.02044.pdf), and
    [JSMA](https://arxiv.org/pdf/1511.07528.pdf). Alternatively, attacks that
    do not "terminate early," return the set of inputs that maximize model
    loss. To be precise, such attacks actually return the set of inputs that
    _improve attack loss_, as this empirically appears to perform (marginally)
    better than measuring model loss. Attacks in this regime include
    [APGD-\*](https://arxiv.org/pdf/2003.01690.pdf),
    [BIM](https://arxiv.org/pdf/1611.01236.pdf), and
    [PGD](https://arxiv.org/pdf/1706.06083.pdf). In the `attacks` module, the
    `update` method of the `Attack` class details the impact of
    `early_termination`. As one use case, it is generally accepted that
    [investigating transferability should be done with attacks configured to
    maximize model loss](https://arxiv.org/pdf/1809.02861.pdf) (i.e.,
    `early_termination` set to `False`).

* Projection: `early_termination` also influences when perturbations are
    projected as to comply with lp-based budgets. With `early_termination`,
    attacks are free to exceed the threat model and the resultant adversarial
    examples at any particular iteration are then projected and compared to the
    best adversarial examples seen thus far (which is _necessary_ for attacks
    that use losses such as [CW Loss](https://arxiv.org/pdf/1608.04644.pdf), as
    attacks that use this loss ostensibly always exceed the threat model and,
    once misclassified, naturally become budget-compliant). However, attacks
    that do not use `early_termination` are always budget-complaint;
    empirically, without enforcing budget-compliance, unbounded adversarial
    examples that maximize model loss are often _worse_ (than continuously
    bounded adversarial examples) with a naïve projection on the last
    iteration.

* Random start: there are may different ways that random starts are
    implemented. For most implementations found online, l∞ attacks that use
    random start initialize perturbations by sampling uniformly between ±ε,
    while perturbations from l2 attacks are sampled from a standard normal
    distribution and subsequently normalized to ε. This repo also supports
    random start for l0 attacks in that an l0-number of features are randomly
    selected per sample, whose values are then sampled uniformly between ±1.

## Repo Overview

This repo is based on [The Space of Adversarial
Strategies](https://arxiv.org/abs/2209.04521). Many of the classes defined
within the various modules are verbatim implementations of concepts introduced
in that paper (with some additions as well). Concisely, the follow components
are described below (and in more detail with the following sections) described
from most abstract to least abstract:

* Adversaries (`attack.py`): control hyperparameter optimization and record
    optimal adversarial examples across multiple runs (useful only if attacks
    have non-deterministic components). Contains an _attack_.
* Attacks (`attack.py`): core attack loop that enforces threat models, domain
    constraints, and keeps track of the best adversarial examples seen
    throughout the crafting process. Contains _travelers_ and _surfaces_.
* Travelers (`traveler.py`): defines techniques that control how data is
    manipulated. Contains _optimizers_ and _random start strategies_.
* Surfaces (`surface.py`): defines techniques that produce and manipulate
    gradients. Contains _losses_, _saliency maps_, and _norms_.
* Optimizers (`optimizer.py`): defines techniques that consume gradient-like
    information and update perturbations. Contains _SGD_, _Adam_, _Backward
    SGD_ (from [FAB](https://arxiv.org/pdf/1907.02044.pdf)), and _Momentum Best
    Start_ (from [APGD-\*](https://arxiv.org/pdf/2003.01690.pdf)).
* Random Starts (`traveler.py`): defines techniques to randomly initialize
    perturbations. Contains _Identity_ (no random start), _Max_
    (from[PGD](https://arxiv.org/pdf/1706.06083.pdf)), and _Shrinking_ (from
    [FAB](https://arxiv.org/pdf/1907.02044.pdf)).
* Losses (`loss.py`): defines measures of error. Contains _Cross-entropy_,
    _Carlini-Wagner_ (from [CW-L2](https://arxiv.org/pdf/1608.04644.pdf)),
    _Difference of Logits Ratio_ (from
    [APGD-DLR](https://arxiv.org/pdf/2003.01690.pdf)), and _Identity_
    (minimizes the model logit associated with the label).
* Saliency Maps (`surface.py`): defines heuristics applied to gradients.
    Contains _DeepFool_ (from [DF](https://arxiv.org/pdf/1511.04599.pdf)),
    _Jacobian_ (from [JSMA](https://arxiv.org/pdf/1511.07528.pdf)), and
    _Identity_ (no saliency map).
* Norms (`surface.py`): manipulates gradients as to operate under the lp-threat
    model. Contains _l0_, _l2_, and _l∞_.

### Adversary

The `Adversary` class (`attacks.py`) serves a wrapper for `Attack` objects.
Specifically, some attacks contain non-deterministic components (such random
initialization, as shown with [PGD](https://arxiv.org/pdf/1706.06083.pdf)), and
thus, the `Adversary` layer records the "best" adversarial examples seen across
multiple runs of an attack (for some definition of "best"). As a second
function, some attacks embed hyperparameter optimization as part of the
adversarial crafting process (such as
[CW-L2](https://arxiv.org/pdf/1608.04644.pdf)). `Adversary` objects are also in
charge of updating hyperparameters across attack runs, based on the success of
resultant adversarial examples.

### Attack

The `Attack` class (`attacks.py`) serves a binder between `Traveler` and
`Surface` objects. Specifically, `Attack` objects perform the standard steps of
any white-box evasion attack in adversarial machine learning: it (1) loops over
a batch of inputs for some number of steps, (2) ensures the resultant
perturbations are both compliant with the parameterized lp budget and feature
ranges for the domain, (3) records various statistics throughout the crafting
process, and (4) keeps track of the "best" adversarial examples seen thus far
(for some definition of "best"). Here, "best" is a function of whether or not
`early_termination` is enabled. For attacks whose goal is to _minimize norm_
(e.g., [JSMA](https://arxiv.org/pdf/1511.07528.pdf)) then an adversarial
example is considered better if it is both misclassified and has smaller norm
than the smallest norm seen. For attacks whose goal is to _maximize (model)
loss_ (e.g., [APGD-CE](https://arxiv.org/pdf/2003.01690.pdf)), then an
adversarial example is considered better if the _attack_ loss has improved
(i.e., the Cross-Entropy loss is higher or the [Difference of Logits
Ratio](https://arxiv.org/pdf/2003.01690.pdf) loss is lower).

### Traveler

The `Traveler` class (`traveler.py`) is one of the two core constructs of any
white-box evasion attack. Specifically, _Travelers_ define techniques that
_manipulate the input_. Fundamentally, optimizers are defined here, in that
they make some informed decision based on gradients (and sometimes additional
information as well). Moreover, random start strategies are also defined here,
in that they initialize the perturbation based on the total budget or the
smallest norm seen thus far.

### Surface

The `Surface` class (`surface.py`) is one of the two core constructs of any
white-box evasion attacks. Specifically, _Surfaces_ defines techniques that
_inform how the input should be manipulated_. Here, loss functions, saliency
maps, and lp norms all manipulate gradients for _Travelers_ to consume.

### Optimizers

Optimizer classes (`optimizer.py`) are part of `Traveler` objects that define
the set of techniques that produce perturbation. They consume gradient
information and apply a perturbation as to maximize (or minimize) the desired
objective function. Four optimizers are currently support: `SGD`, `Adam`,
`BackwardSGD`, and `MomentumBestStart`. `SGD` and `Adam` are common optimizers
used in machine learning and are simply imported into the `optimizer` module
namespace from PyTorch. `BackwardSGD` comes from
[FAB](https://arxiv.org/pdf/1907.02044.pdf); specifically, it (1) performs a
backward step for misclassified inputs (as to minimize perturbation norm), and
(2) performs a biased projection towards the original input (also to minimize
perturbation norm) with a standard update step in direction of the gradients.
`MomentumBestStart` comes from [APGD-\*](https://arxiv.org/pdf/2003.01690.pdf);
specifically, it measures the progress of perturbations at a series of
checkpoints. If progress has stalled (measured by a stagnating increase (or
decrease) in attack loss), then the perturbation is reset to the best seen
perturbation and the learning rate is halved. Conceptually, `MomentumBestStart`
starts with aggressive perturbations (the learning rate is initialized to ε for
l2 and l∞ attacks and 1.0 for l0 attacks) and iteratively refines it when a
finer search is warranted.

### Random Start Strategies

Random start strategies (`traveler.py`) are part of `Traveler` objects that
define the set of techniques used to initialize perturbations. They either
initialize perturbations by randomly sampling within the budget or based on the
norm of the best perturbations seen thus far. Three random start strategies are
supported: `MaxStart`, `Identity`, and `ShrinkingStart`. `IdentityStart` serves
as a "no random start" option—the input is returned as-is. `MaxStart` comes
from [PGD](https://arxiv.org/pdf/1706.06083.pdf); specifically, it initializes
perturbations randomly based on the perturbation budget. `ShrinkingStart` comes
from [FAB](https://arxiv.org/pdf/1907.02044.pdf); specifically, it initializes
perturbations based on minimum of the best perturbation seen thus far (where
"best" is defined as the smallest perturbation vector that was still
misclassified) and the perturbation budget. When paired with `Adversary`
restart capabilities, `ShrinkingStart` initially performs like `MaxStart`, and
gradually produces smaller initializations for a finer search.

### Losses

Loss functions (`loss.py`) are part of `Surface` objects that define measures
of error. When differentiated, they inform how perturbations should be
manipulated such that adversarial goals are met. Four losses are supported:
`CELoss`, `CWLoss`, `DLRLoss`, and `IdentityLoss`. `CELoss` is perhaps the most
popular loss function used in attacks, given its popularity in training deep
learning models, and is a simple wrapper for `torch.nn.CrossEntropyLoss`.
`CWLoss` comes from [CW-L2](https://arxiv.org/pdf/1608.04644.pdf);
specifically, it measures the difference of the logits associated with the
label and the next closest class and the current l2-norm of the perturbation.
`DLRLoss` comes from [APGD-DLR](https://arxiv.org/pdf/2003.01690.pdf);
specifically, it also measures the difference of logits associated with the
label and the next closest class, normalized by the difference of the largest
and third-largest logic (principally used to prevent vanishing gradients).
`IdentityLoss` serves as the "lack" of a loss function in that it just returns
the model logit associated with the label.

### Saliency Maps

Saliency maps (`surface.py`) are part of `Surface` objects that apply
heuristics to gradients to help meet adversarial goals. Three saliency maps are
supported: `DeepFoolSaliency`, `IdentitySaliency`, and `JacobianSaliency`.
`DeepFoolSaliency` comes from [DF](https://arxiv.org/pdf/1511.04599.pdf);
specifically, it approximates the projection of the input onto the decision
manifold. `IdentitySaliency` serves as a "no saliency map" option in that the
gradients are returned as-is. `JacobianSaliency` comes from
[JSMA](https://arxiv.org/pdf/1511.07528.pdf); specifically, it scores features
based on how perturbing them will simultaneously move inputs away from their
labels and towards other classes.

### Norms

Norms (`surface.py`) are part of `Surface` objects that projects gradients into
the lp-norm space of the attack. Three norms are supported: `L0`, `L2`, and
`Linf`. `L0` comes from [JSMA](https://arxiv.org/pdf/1511.07528.pdf);
specifically, this computes the gradient component with largest magnitude and
sets all other components to zero. `L2` comes from
[CW-L2](https://arxiv.org/pdf/1608.04644.pdf),
[DF](https://arxiv.org/pdf/1511.04599.pdf), and
[FAB](https://arxiv.org/pdf/1907.02044.pdf); specifically, this normalizes the
gradients by their l2-norm. `Linf` comes from
[APGD-CE](https://arxiv.org/pdf/2003.01690.pdf),
[APGD-DLR](https://arxiv.org/pdf/2003.01690.pdf),
[BIM](https://arxiv.org/pdf/1611.01236.pdf), and
[PGD](https://arxiv.org/pdf/1706.06083.pdf); specifically, this returns the
sign of the gradients.

Norm objects also serve a dual purpose in that they are also called by `Attack`
objects to ensure perturbations are compliant with the parameterized lp budget.
`L0` objects project onto the l0-norm by computing the top-ε (by magnitude)
perturbation components (where ε defines number of perturbable features) and
setting all other components zero. `L2` objects project onto the l2-norm by
renorming perturbations such that their l2-norm is no greater than the budget.
`Linf` objects project onto the l∞-norm by ensuring the value of perturbation
component is between -ε and ε (where ε defines the maximum allowable change
across all features).

## Parameters

While this repo uses sane defaults for the many parameters used in attacks, the
core initialization parameters are listed here for reference, categorized by
class.

### Adversary

* `best_update`: update rule to determine if an adversarial example is "better"
* `hparam`: hyperparameter to optimize with binary search
* `hparam_bounds`: initial bounds when using binary search for hyperparameters
* `hparam_steps`: the number of binary search steps
* `hparam_update`: update rule to determine if a hyperparameter should be
increased (or decreased)

### Attack

* `alpha`: perturbation strength per iteration
* `early_termination`: whether to stop perturbing adversarial examples as soon
as they are misclassified
* `epochs`: number of steps to compute perturbations
* `epsilon`: lp budget
* `loss_func`: loss function to use
* `norm`: lp-norm to use
* `model`: reference to a [deep learning model](https://github.com/sheatsley/models)
* `optimizer_alg`: optimizer to use
* `saliency_map`: saliency map to use

### Traveler

* `optimizer`: optimizer to use
* `random_start`: random start strategy to use

### Surface

* `loss`: loss function to use
* `model`: reference to a [deep learning model](https://github.com/sheatsley/models)
* `norm`: lp-norm to use
* `saliency_map`: saliency map to use

### Optimizer

#### Adam

* [The implementation of Adam in PyTorch is used verbatim](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

#### BackwardSGD

* `params`: perturbations
* `attack_loss`: attack loss (caches model accuracy on forward passes)
* `lr`: perturbation strength per iteration
* `maximize`: whether the attack loss is to be maximized (or minimized)
* `norm`: lp-norm used
* `smap`: saliency map used (caches biased projection when using DeepFool)
* `alpha_max`: maximum strength of biased projection
* `beta`: backward step strength for misclassified inputs

#### MomentumBestStart

* `params`: perturbations
* `attack_loss`: attack loss (caches attack loss on forward passes)
* `epochs`: total number of optimization iterations
* `epsilon`: lp budget
* `maximize`: whether the attack loss is to be maximized (or minimized)
* `alpha`: momentum factor
* `pdecay`: period length decay
* `pmin`: minimum period length
* `rho`: minimum percentage of successful updates between checkpoints

#### SGD

* [The implementation of SGD in PyTorch is used verbatim](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

### Random Start Strategies

#### IdentityStart

* No parameters are necessary for this class

#### MaxStart

* `norm`: lp-norm used
* `epsilon`: lp budget

#### ShrinkingStart

* `norm`: lp-norm used
* `epsilon`: lp budget

### Loss

#### CELoss

* No parameters are necessary for this class

#### CWLoss

* `classes`: number of classes
* `c`: initial value weighing the influence of misclassification over norm
* `k`: desired logit difference

#### DLRLoss

* `classes`: number of classes

#### IdentityLoss

* No parameters are necessary for this class

### Saliency Maps

#### DeepFoolSaliency

* `p`: lp-norm used
* `classes`: number of classes

#### IdentitySaliency

* No parameters are necessary for this class

#### JacobianSaliency

* No parameters are necessary for this class

### Norms

#### L0

* `epsilon`: maximum l0 distance for perturbations
* `maximize`: whether the attack loss is to be maximized (or minimized)

#### L2

* `epsilon`: maximum l2 distance for perturbations

#### Linf

* `epsilon`: maximum l∞ distance for perturbations

### Misc

Here are a list of some subtle parameters you may be interested in manipulating
for a deeper exploration:

* `betas`, `eps`, `weight_decay` and `amsgrad` in `Adam` optimizer
* `alpha_max`, `beta`, and `minimum` in `BackwardSGD` optimizer
* `alpha`, `pdecay`, `pmin`, and `rho` in `MomentumBestStart` optimizer
* `minimum` in `MaxStart` random start strategy
* `c` and `k` in `CWLoss` loss
* `minimum` in `DLRLoss` loss
* `minimum` in `DeepFoolSaliency` saliency map
* `top` in `L0` norm
* `minimum` in `L2` norm

Moreover, below are some attack-specific observations:

* The `DeepFoolSaliency` saliency map computes (in some sense) step sizes
dynamically (Specifically, the absolute value of the logit differences over the
normed gradient differences). Thus, when paired with an optimizer that lacks an
adaptive learning rate (i.e., `BackwardSGD` and `SGD`), `alpha` in `Attack`
objects should be set to `1.0`. This is done by default when instantiating
`Attack` objects and can be overridden by setting `alpha_override` to `False`.

* When using attacks with non-deterministic components (e.g., random start
strategies) or hyperparameters (e.g., `CWLoss`), leveraging the `Adversary`
layer to repeat attacks multiple times or optimize hyperparameters can be an
effective strategy (`hparam_bounds`, `hparam_steps`, and `num_restarts` are the
parameters of interest in this regime).

## Citation

You can cite this repo as follows:

    @misc{https://doi.org/10.48550/arxiv.2209.04521,
      doi = {10.48550/ARXIV.2209.04521},
      url = {https://arxiv.org/abs/2209.04521},
      author = {Sheatsley, Ryan and Hoak, Blaine and Pauley, Eric and McDaniel, Patrick},
      keywords = {Cryptography and Security (cs.CR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {The Space of Adversarial Strategies},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }
