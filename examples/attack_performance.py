"""
This script compares the performance of known attacks within the aml framework
and plots the model accuracy and loss over the crafting epochs.
"""
import argparse
import warnings

import aml
import dlm
import mlds
import pandas
import seaborn
import torch

# dlm uses lazy modules which induce warnings that overload stdout
warnings.filterwarnings("ignore", category=UserWarning)


def main(alpha, attacks, budget, datasets, epochs, trials):
    """
    This function is the main entry point for the performance comparison
    benchmark. Specifically, this: (1) loads and trains a model for each
    dataset, (2) crafts adversarial examples with each attack, (3) measures
    statistics, and (5) plots the results.

    :param alpha: perturbation strength, per-iteration
    :type alpha: float
    :param attacks: attacks to use
    :type attacks: list of str
    :param budget: maximum lp budget
    :type budget: float
    :param datasets: dataset(s) to use
    :type datasets: tuple of str
    :param epochs: number of attack iterations
    :type epochs: int
    :param trials: number of experiment trials
    :type trials: int
    :return: None
    :rtype: NoneType
    """
    print(
        f"Analyzing {len(attacks)} attacks across {len(datasets)} datasets "
        f"with {trials} trials..."
    )
    results = pandas.DataFrame(
        columns=("dataset", "attack", "budget", "epoch", "data", "performance")
    )

    # load data, train a model, and compute budgets
    for i, d in enumerate(datasets):
        data = getattr(mlds, d)
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
        template = getattr(dlm.templates, d)
        model = (
            dlm.CNNClassifier(**template.cnn)
            if hasattr(template, "cnn")
            else dlm.MLPClassifier(**template.mlp)
        )
        model.verbosity = 0
        for t in range(trials):
            print(
                f"On dataset {i + 1} ({d}) of {len(datasets)}, trial {t + 1} of "
                f"{trials}... ({(t + i * trials) / (len(datasets) * trials):.2%})"
            )
            model.fit(train_x, train_y)
            l0 = int(x.size(1) * budget) + 1
            l2 = (
                x.max(0)
                .values.clamp(min=1)
                .sub(x.min(0).values.clamp(max=0))
                .norm(2)
                .item()
                * budget
            )
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
            for a in attacks:
                attack = a(alpha, epochs, norms[a], model, statistics=True)
                attack.craft(x, y)
                attack_results = attack.res.assign(
                    dataset=d,
                    attack=attack.name,
                    budget=getattr(attack.res, attack.norm_class.__name__.lower())
                    / norms[a],
                ).melt(
                    id_vars=("dataset", "attack", "budget", "epoch"),
                    value_vars=("accuracy", "model_loss"),
                    var_name="data",
                    value_name="performance",
                )
                results = pandas.concat((results, attack_results))

    # plot results and save
    results.replace("model_loss", "loss")
    plot(results)
    return None


def plot(results):
    """
    This function plots the attack comparison results. Specifically, this
    produces two line plots per dataset containing model accuracy and loss over
    the attack epoch, with shading represtenting 95% CI. Attacks are divided by
    color. The plot is written to disk in the current directory.

    :param results: results of the attack comparison
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    plot = seaborn.relplot(
        data=results,
        col="dataset",
        facet_kws=dict(sharex=False, sharey=False),
        hue="attack",
        kind="line",
        legend="full" if results.dataset.unique().size > 1 else "auto",
        row="data",
        x="epoch",
        y="performance",
    )
    plot.savefig(__file__[:-3] + ".pdf", bbox_inches="tight")
    return None


if __name__ == "__main__":
    """
    This script compares the performance of known attacks in the aml library.
    Datasets are provided by mlds (https://github.com/sheatsley/datasets) and
    models by dlm (https://github.com/sheatsley/models). Specifically, this
    script: (1) parses command-line arguments, (2) loads dataset(s), (3) trains
    a model, (4) crafts adversarial examples with each attack, (5) collects
    statistics on model accuracy and lp-norm of the adversarial examples, and
    (6) plots the results.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial machine learning attack comparison"
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
        choices=mlds.__available__,
        default=mlds.__available__,
        help="Dataset(s) to use",
        nargs="+",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        help="Number of attack iterations",
        type=int,
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        help="Number of experiment trials",
        type=int,
    )
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        attacks=args.attacks,
        budget=args.budget,
        datasets=args.datasets,
        epochs=args.epochs,
        trials=args.trials,
    )
    raise SystemExit(0)
