"""
This script visualizes the worst-, average-, and best-case perturbations for
known attacks within the aml framework.
Author: Ryan Sheatsley
Thu Mar 9 2023
"""
import argparse

import aml
import dlm
import matplotlib.pyplot as plt
import mlds
import pandas
import torch


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
        nrows=len(results),
        ncols=3,
        subplot_kw=dict(frameon=False, xticks=[], yticks=[]),
    )
    plot.suptitle(dataset)
    for i, axis in enumerate(axes.flat):
        attack, perturbation = results.loc[i]
        axis.imshow(perturbation, cmap="gray")
        if i == 0:
            axis.set_text("Minimum Perturbation")
        elif i == 1:
            axis.set_text("Average Perturbation")
        elif i == 2:
            axis.set_text("Maximum Perturbation")
        if i % 3 == 0:
            axis.set_ylabel(attack, rotation=0)
    plot.tight_layout()
    plot.savefig(__file__[:-2] + "pdf", bbox_inches="tight")
    return None


def main(alpha, attacks, budget, dataset, epochs):
    """
    This function is the main entry point for the perturbation visualization.
    Specifically, this: (1) loads and trains a model for each dataset, (2)
    crafts adversarial examples with each attack, (3) measures perturbation
    statistics, and (5) plots the results.

    :param alpha: perturbation strength, per-iteration
    :type alpha: float
    :param attacks: attacks to use
    :type attacks: list of str
    :param budget: maximum lp budget
    :type budget: float
    :param dataset: dataset to use
    :type dataset: str
    :param epochs: number of attack iterations
    :type epochs: int
    :return: None
    :rtype: NoneType
    """

    # load data, train a model, and compute budgets
    print(f"Attacking {dataset} with {len(attacks)} attacks..")
    results = pandas.DataFrame(columns=("attack", "perturbation"))
    data = getattr(mlds, dataset)
    try:
        train_x = torch.from_numpy(data.train.data)
        train_y = torch.from_numpy(data.train.labels).long()
        x = torch.from_numpy(data.test.data)
        y = torch.from_numpy(data.test.labels).long()
    except AttributeError:
        train_x = torch.from_numpy(data.dataset.data)
        train_y = torch.from_numpy(data.dataset.labels).long()
        x = train_x.clone()
        y = train_y.clone()
    template = getattr(dlm.templates, dataset)
    model = dlm.CNNClassifier(**template.cnn)
    model.fit(train_x, train_y)
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
        print(f"Attacking with {a}... ({i} of {len(attacks)})")
        attack = a(alpha, epochs, norms[a], model)
        p = attack.craft(x, y)

        # find worst-, average-, and best-case perturbations
        p = p[(p > 0).any(1)]
        _, p_idx = p.norm(norms[a], 1).sort().values
        idxs = [p_idx[0].item(), p_idx[p_idx.numel() // 2].item(), p_idx[-1].item()]
        min_adv, med_adv, max_adv = (x[idxs] + p[idxs]).reshape(-1, size, size)
        for j, adv in enumerate(min_adv, med_adv, max_adv):
            results[j + 3 * i] = attack.name, adv

    # plot results and save
    plot(dataset, results)
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
        "--datasets",
        choices=(
            d for d in mlds.__available__ if hasattr(getattr(dlm.templates, d), "cnn")
        ),
        default="mnist",
        help="Dataset to use",
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
        datasets=args.datasets,
        epochs=args.epochs,
    )
    raise SystemExit(0)
