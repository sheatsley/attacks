# Adversarial Machine Learning

_Adversarial Machine Learning_ (`aml`) is a repo for measuring  the robustness
of deep learning models against white-box evasion attacks. Designed for
academics, it is principally designed for use in fundamental research to
understand _adversarial examples_, inputs designed to cause models to make a
mistake[[1](https://arxiv.org/abs/1412.6572)]. At its core, `aml` is based
on a series of techniques used in six popular attacks:

1. [APGD-CE](https://arxiv.org/pdf/2003.01690.pdf) (Auto-PGD with CE loss)
2. [APGD-DLR](https://arxiv.org/pdf/2003.01690.pdf) (Auto-PGD with DLR loss)
3. [BIM](https://arxiv.org/pdf/1611.01236.pdf) (Basic Iterative Method)
4. [CW-L2](https://arxiv.org/pdf/1608.04644.pdf) (Carlini-Wagner with l₂ norm)
5. [DF](https://arxiv.org/pdf/1511.04599.pdf) (DeepFool)
6. [FAB](https://arxiv.org/pdf/1907.02044.pdf) (Fast Adaptive Boundary)
7. [JSMA](https://arxiv.org/pdf/1511.07528.pdf) (Jacobian Saliency Map Approach)
8. [PGD](https://arxiv.org/pdf/1706.06083.pdf) (Projected Gradient Descent)

I emphasize that that components defined here are based on these attacks
because I have taken certain liberties in modifying the implementation of these
techniques to either improve their performance or their clarity (without any
cost in performance). Not only are these modifications designed to help
academics understand _why_ attacks perform the way that they do, but also to
[serve as abstractions for effortlessly building a vast space of new
attacks](https://arxiv.org/abs/2209.04521). At this time, the techniques based
on the eight attacks above enable construction of 432 total attacks (all of
which often decrease model accuracy to less than 1% with ~100 iterations across
the datasets found in [this repo](https://github.com/sheatsley/datasets)). All
of the information you need to start using this repo is contained within this
one ReadMe, ordered by complexity (No need to parse through some highly
over-engineered ReadTheDocs documentation).

## Table of Contents

* [Quick start](#quick-start)
* [Repo Overview](#library-overview)
* [Hyperparameters](#hyperparameters)
* [Citation](#citation)

## Quick start

This repo is, by design, to be interoperable with the following
[datasets](https://github.com/sheatsley/datasets) and
[models](https://github.com/sheatsley/models) repos (which are all based on
[PyTorch](https://github.com/pytorch/pytorch)). With some effort, you
probably bring your own data and models, but I wouldn't recommend it if you're
looking to start using this repo as easily as possible. I recommend installing
an editable version of this repo via `pip install -e`. Afterwards, you can
craft adversarial examples using any of the eight attacks above as follows:

```
import aml # ML robustness evaluations with PyTorch
import dlm  # Pytorch-based deep learning models with scikit-learn-like interfaces
import mlds  # Scripts for downloading, preprocessing, and numpy-ifying popular machine learning datasets
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# load data
mnist = mlds.mnist
x_train = torch.from_numpy(mnist.train.data)
y_train = torch.from_numpy(mnist.train.labels).long()
x_test = torch.from_numpy(mnist.test.data)
y_test = torch.from_numpy(mnist.test.labels).long()

# instantiate and train a model
hyperparameters = dlm.hyperparameters.mnist
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
```

## Repo Overview

This repo is based on [The Space of Adversarial
Strategies](https://arxiv.org/abs/2209.04521). Many of the classes defined
within the various modules are verbatim implementations of concepts introduced
in that paper (with some additions; I use this repo for ostensibly anything
related to adversarial machine learning).

## Hyperparameters

## Citation

You can cite this repo as follows:
```
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
```
