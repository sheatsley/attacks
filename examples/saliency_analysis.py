"""
This script performs a perturbation saliency analysis of known attacks within
the aml framework and plots the distribution of perturbations, per-class.
"""
import argparse

import aml
import dlm
import matplotlib
import mlds
import pandas
import seaborn
import torch


def plot(dataset, results):
    """
    This function plots the perturbation saliency results. Specifically, this
    produces n scatter plots (where n is the number of attacks) of the mean of
    the perturbations with classes on the y-axis and the feature number on the
    x-axis. Perturbation values are encoded by hue and error bars represent the
    standard deviation. The plot is written to disk in the current directory.

    :param dataset: dataset used
    :type dataset: str
    :param results: results of the saliency analysis
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """

    # https://github.com/mwaskom/seaborn/blob/a69eb55f2d89b00e1d31b9c9ec29982fe0a187cb/seaborn/regression.py#L397
    seaborn.set_style("whitegrid")
    matplotlib.rcParams["lines.linewidth"] = 0.01
    plot = seaborn.lmplot(
        data=results,
        col="attack",
        col_wrap=(results.attack.unique().size + 1) // 2,
        facet_kws=dict(sharey=False),
        fit_reg=False,
        hue="class",
        legend="full" if results.attack.unique().size > 1 else "auto",
        scatter_kws=dict(alpha=0.8, facecolors="none", linewidths=0.5, s=10),
        x="feature",
        y="value",
        x_ci="sd",
        x_bins=int(results["feature"].max() + 1),
        x_jitter=1 / results["class"].max(),
        y_jitter=1 / results["class"].max(),
    )
    plot.fig.suptitle(f"dataset = {dataset}")
    plot.savefig(__file__[:-3] + f"_{dataset}.pdf")
    return None


def main(alpha, attacks, budget, dataset, epochs):
    """
    This function is the main entry point for the saliency analysis.
    Specifically, this: (1) loads and trains a model for each dataset, (2)
    crafts adversarial examples with each attack, and (3) plots the results.

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
    results = pandas.DataFrame(columns=("attack", "class", "feature", "value"))
    model = (
        dlm.CNNClassifier(**template.cnn)
        if hasattr(template, "cnn")
        else dlm.MLPClassifier(**template.mlp)
    )
    model.fit(train_x, train_y)
    print(f"Crafting set: ({x.size(0), x.size(1)}) ✕ ({y.numel()},)")
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

    # instantiate attacks, craft adversarial examples, and collect results
    labels = y.unsqueeze(-1).expand(x.size()).flatten()
    features = torch.arange(x.size(1)).unsqueeze(0).expand(x.size()).flatten()
    for i, a in enumerate(attacks):
        print(f"Attacking with {a.__name__}... ({i} of {len(attacks)})")
        attack = a(alpha, epochs, norms[a], model)
        p = attack.craft(x, y).flatten()

        # exclude features that were not perturbed
        attack_results = torch.vstack((labels, features, p))[:, p.nonzero().flatten()].T
        res = pandas.DataFrame(attack_results, columns=("class", "feature", "value"))
        res["attack"] = attack.name
        results = pandas.concat((results, res))

    # plot results and save
    plot(dataset, results.reset_index())
    return None


if __name__ == "__main__":
    """
    This script performs a perturbation saliency analysis of known attacks
    within the aml library. Datasets are provided by mlds
    (https://github.com/sheatsley/datasets) and models by dlm
    (https://github.com/sheatsley/models). Specifically, this script: (1)
    parses command-line arguments, (2) loads dataset(s), (3) trains a model,
    (4) crafts adversarial examples with each attack, (5) separates
    perturbations for each attack by label, and (6) plots the results.
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
        choices=mlds.__available__,
        default="phishing",
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
        dataset=args.dataset,
        epochs=args.epochs,
    )
    raise SystemExit(0)
