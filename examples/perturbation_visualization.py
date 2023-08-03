"""
This script visualizes the worst-, average-, and best-case perturbations for
known attacks within the aml framework.
"""
import argparse

import aml
import dlm
import matplotlib.pyplot as plt
import mlds
import torch


def main(alpha, attacks, budget, dataset, device, epochs):
    """
    This function is the main entry point for the perturbation visualization.
    Specifically, this: (1) loads and trains a model, (2) crafts adversarial
    examples with each attack, (3) measures perturbation statistics, and (5)
    plots the results.

    :param alpha: perturbation strength, per-iteration
    :type alpha: float
    :param attacks: attacks to use
    :type attacks: list of str
    :param budget: maximum lp budget
    :type budget: float
    :param dataset: dataset to use
    :type dataset: str
    :param device: hardware device to use
    :type device: str
    :param epochs: number of attack iterations
    :type epochs: int
    :return: None
    :rtype: NoneType
    """

    # load data, train a model, and compute budgets
    print(f"Attacking {dataset} with {len(attacks)} attacks..")
    data = getattr(mlds, dataset)
    results = []
    try:
        xt = torch.from_numpy(data.train.data).to(device)
        yt = torch.from_numpy(data.train.labels).long().to(device)
        x = torch.from_numpy(data.test.data).to(device)
        y = torch.from_numpy(data.test.labels).long().to(device)
    except AttributeError:
        xt = torch.from_numpy(data.dataset.data).to(device)
        yt = torch.from_numpy(data.dataset.labels).long().to(device)
        x = xt
        y = yt
    template = getattr(dlm.templates, dataset)
    model = dlm.CNNClassifier(**template.cnn | dict(device=device))
    model.fit(xt, yt)
    size = int(x.size(1) ** (1 / 2))
    l0 = int(x.size(1) * budget) + 1
    l2 = size * budget
    linf = budget
    norms = {
        aml.apgdce: linf,
        aml.apgddlr: linf,
        aml.bim: linf,
        aml.cwl2: l2,
        aml.df: l2,
        aml.fab: l2,
        aml.jsma: l0,
        aml.pgd: linf,
    }

    # instantiate attacks and craft adversarial examples
    for i, a in enumerate(attacks):
        print(f"Attacking with {a.__name__}... ({i} of {len(attacks)})")
        attack = a(alpha, epochs, norms[a], model, verbosity=0)
        p = attack.craft(x, y)

        # find worst-, average-, and best-case perturbations
        successful = (p > 0).any(1)
        p = p[successful]
        xs = x[successful]
        p_idx = p.norm(attack.lp, 1).sort().indices
        idxs = [p_idx[0].item(), p_idx[p_idx.numel() // 2].item(), p_idx[-1].item()]
        results.append(
            (a.__name__, *(xs[idxs] + p[idxs]).reshape(-1, size, size).cpu().unbind(0))
        )

    # plot results and save
    plot(dataset, results)
    return None


def plot(dataset, results):
    """
    This function visualizes the perturbation results. Specifially, this
    produces a 3-by-n (where n is the number of attacks) grid visualizing the
    worst-, average-, and best-case perturbations (as measured by lp-norm).
    Attacks are separated by rows with perturbation types by column. The plot
    is written to disk in the current directory.

    :param dataset: dataset used
    :type dataset: str
    :param results: results of the attack comparison
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    plot, axes = plt.subplots(
        figsize=(5, 1.5 * len(results)),
        ncols=3,
        nrows=len(results),
        squeeze=False,
        subplot_kw=dict(frameon=False, xticks=[], yticks=[]),
    )
    axes[0, 0].set_title("minimum")
    axes[0, 1].set_title("average")
    axes[0, 2].set_title("maximum")
    plot.suptitle(f"dataset={dataset}")
    for idx, result in enumerate(results):
        axes[idx, 0].set_ylabel(result[0], labelpad=20, rotation=0)
        for idy, img in enumerate(result[1:]):
            axes[idx, idy].imshow(img, cmap="gray")
    plot.tight_layout()
    plot.savefig(__file__[:-3] + f"_{dataset}.pdf", bbox_inches="tight")
    return None


if __name__ == "__main__":
    """
    This script visualizes perturbartions for known attacks in the aml library.
    Datasets are provided by mlds (https://github.com/sheatsley/datasets) and
    models by dlm (https://github.com/sheatsley/models). Specifically, this
    script: (1) parses command-line arguments, (2) loads dataset(s), (3) trains
    a model, (4) crafts adversarial examples with each attack, (5) computes the
    worst-, average-, and best-case perturbations (as measured by lp norm), and
    (6) plots the results.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial machine learning  perturbation visualization"
    )
    parser.add_argument(
        "--alpha",
        default=0.01,
        help="Perturbation strength, per-iteration",
        type=float,
    )
    parser.add_argument(
        "-a",
        "--attacks",
        choices=(
            aml.apgdce,
            aml.apgddlr,
            aml.bim,
            aml.cwl2,
            aml.df,
            aml.fab,
            aml.jsma,
            aml.pgd,
        ),
        default=(
            aml.apgdce,
            aml.apgddlr,
            aml.bim,
            aml.cwl2,
            aml.df,
            aml.fab,
            aml.jsma,
            aml.pgd,
        ),
        help="Attacks to use",
        nargs="+",
        type=lambda a: getattr(aml, a),
    )
    parser.add_argument(
        "-b",
        "--budget",
        default=0.15,
        help="Maximum lp budget (as a percent)",
        type=float,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=(
            d for d in mlds.__available__ if hasattr(getattr(dlm.templates, d), "cnn")
        ),
        default="mnist",
        help="Dataset to use",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Hardware device to use",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        help="Number of attack iterations",
        type=int,
    )
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        attacks=args.attacks,
        budget=args.budget,
        dataset=args.dataset,
        device=args.device,
        epochs=args.epochs,
    )
    raise SystemExit(0)
