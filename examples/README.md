# Adversarial Machine Learning Examples

This directory contains various examples showing how `aml` can be used. Note
that these examples will write figures to this directory. All of these examples
depend on [dlm](https://github.com/sheatsley/models)
[mlds](https://github.com/sheatsley/datasets) repos and investigate the
following known attacks:[APGD-CE](https://arxiv.org/pdf/2003.01690.pdf),
[APGD-DLR](https://arxiv.org/pdf/2003.01690.pdf),
[BIM](https://arxiv.org/pdf/1611.01236.pdf),
[CW-L2](https://arxiv.org/pdf/1608.04644.pdf),
[DF](https://arxiv.org/pdf/1511.04599.pdf),
[FAB](https://arxiv.org/pdf/1907.02044.pdf),
[JSMA](https://arxiv.org/pdf/1511.07528.pdf), and
[PGD](https://arxiv.org/pdf/1706.06083.pdf).


* `attack_performance.py`: compares attack performance from known attacks
    and plots model accuracy and loss over epochs on validation data.
* `framework_comparison.py`: compares attack performance of known attacks
    across implementations from
    [AdverTorch](https://github.com/BorealisAI/advertorch),
    [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox),
    [CleverHans](https://github.com/cleverhans-lab/cleverhans),
    [Foolbox](https://github.com/bethgelab/foolbox), &
    [Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch),
    and plots model accuracy over budget consumed.
* `perturbation_visualization.py`: visualizes perturbations of known attacks
    and plots one example of a worst-, average-, and best-case perturbation (as
    measured by lp norm).
