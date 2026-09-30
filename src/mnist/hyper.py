from pathlib import Path
import csv

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Hyperparameter experiments
#
# 這一階段只調 SGDClassifier。
# 不使用 Normalization、Standardization、HOG、Augmentation。
# ============================================================

EXPERIMENTS = [
    # --------------------------------------------------------
    # Original baseline
    # sklearn SGDClassifier default-like setting
    # --------------------------------------------------------
    {
        "name": "baseline",
        "params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 1e-4,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    # --------------------------------------------------------
    # alpha
    # regularization strength
    # --------------------------------------------------------
    {
        "name": "alpha_1e-5",
        "params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 1e-5,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    {
        "name": "alpha_1e-3",
        "params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 1e-3,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    # --------------------------------------------------------
    # loss
    # --------------------------------------------------------
    {
        "name": "log_loss",
        "params": {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    {
        "name": "modified_huber",
        "params": {
            "loss": "modified_huber",
            "penalty": "l2",
            "alpha": 1e-4,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    # --------------------------------------------------------
    # penalty
    # --------------------------------------------------------
    {
        "name": "penalty_l1",
        "params": {
            "loss": "hinge",
            "penalty": "l1",
            "alpha": 1e-4,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    {
        "name": "penalty_elasticnet",
        "params": {
            "loss": "hinge",
            "penalty": "elasticnet",
            "alpha": 1e-4,
            "average": False,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    # --------------------------------------------------------
    # Averaged SGD
    # --------------------------------------------------------
    {
        "name": "average_true",
        "params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 1e-4,
            "average": True,
            "max_iter": 1000,
            "tol": 1e-3,
        },
    },

    # --------------------------------------------------------
    # More iterations
    # --------------------------------------------------------
    {
        "name": "max_iter_2000",
        "params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 1e-4,
            "average": False,
            "max_iter": 2000,
            "tol": 1e-3,
        },
    },
]


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"
    output_dir = PROJECT_ROOT / "outputs"

    output_dir.mkdir(exist_ok=True)

    # ========================================================
    # Load original MNIST training data
    # ========================================================
    x = np.load(
        data_dir / "x.npy",
        mmap_mode="r"
    )[:60000]

    y = np.load(
        data_dir / "y.npy",
        mmap_mode="r"
    )[:60000]

    print("Dataset shape:", x.shape)
    print("Dataset dtype:", x.dtype)

    print()
    print(
        "No normalization / standardization / "
        "HOG / augmentation"
    )

    # ========================================================
    # Same CV setting as original baseline:
    #
    # cross_val_score(..., cv=3)
    # -> StratifiedKFold(n_splits=3, shuffle=False)
    # ========================================================
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=False
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    results = []

    print()
    print(
        "Experiment           | "
        "Accuracy | "
        "Macro F1"
    )

    print("-" * 52)

    # ========================================================
    # Run experiments
    # ========================================================
    for experiment in EXPERIMENTS:
        name = experiment["name"]
        params = experiment["params"]

        model = SGDClassifier(
            random_state=42,

            # Cross-validation 本身已平行化，
            # 避免 nested parallelism
            n_jobs=1,

            **params,
        )

        scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring=scoring,

            # 3 folds 同時跑
            n_jobs=3,
            pre_dispatch=3,

            error_score="raise",
        )

        fold_accuracy = scores["test_accuracy"]
        fold_f1 = scores["test_macro_f1"]

        mean_accuracy = fold_accuracy.mean()
        mean_f1 = fold_f1.mean()
        accuracy_std = fold_accuracy.std()

        results.append(
            {
                "name": name,
                "fold1": fold_accuracy[0],
                "fold2": fold_accuracy[1],
                "fold3": fold_accuracy[2],
                "accuracy": mean_accuracy,
                "macro_f1": mean_f1,
                "accuracy_std": accuracy_std,
                "params": params,
            }
        )

        print(
            f"{name:20s} | "
            f"{mean_accuracy:8.4%} | "
            f"{mean_f1:8.4%}"
        )

        print(
            "    folds:",
            " | ".join(
                f"{score:.4%}"
                for score in fold_accuracy
            )
        )

    # ========================================================
    # Find baseline
    # ========================================================
    baseline = next(
        result
        for result in results
        if result["name"] == "baseline"
    )

    baseline_accuracy = baseline["accuracy"]
    baseline_f1 = baseline["macro_f1"]

    # ========================================================
    # Ranking
    # ========================================================
    results.sort(
        key=lambda result: result["accuracy"],
        reverse=True
    )

    print()
    print("=" * 72)
    print("Ranking")
    print("=" * 72)

    for rank, result in enumerate(
        results,
        start=1
    ):
        accuracy_change = (
            result["accuracy"]
            - baseline_accuracy
        ) * 100

        f1_change = (
            result["macro_f1"]
            - baseline_f1
        ) * 100

        print(
            f"{rank:2d}. "
            f"{result['name']:20s} "
            f"Acc={result['accuracy']:.4%} "
            f"F1={result['macro_f1']:.4%} "
            f"ΔAcc={accuracy_change:+.3f} pp "
            f"ΔF1={f1_change:+.3f} pp"
        )

    # ========================================================
    # Best configuration
    # ========================================================
    best = results[0]

    print()
    print("=" * 72)
    print("Best configuration")
    print("=" * 72)

    print(
        "Experiment:",
        best["name"]
    )

    print(
        f"Accuracy: "
        f"{best['accuracy']:.4%}"
    )

    print(
        f"Macro F1: "
        f"{best['macro_f1']:.4%}"
    )

    print(
        f"Accuracy std: "
        f"{best['accuracy_std']:.4%}"
    )

    print()
    print("Parameters:")

    for key, value in best["params"].items():
        print(
            f"  {key}: {value}"
        )

    # ========================================================
    # Compare best with original baseline
    # ========================================================
    accuracy_improvement = (
        best["accuracy"]
        - baseline_accuracy
    ) * 100

    f1_improvement = (
        best["macro_f1"]
        - baseline_f1
    ) * 100

    print()
    print("=" * 72)
    print("Improvement over baseline")
    print("=" * 72)

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4%}"
    )

    print(
        f"Best accuracy:     "
        f"{best['accuracy']:.4%}"
    )

    print(
        f"Accuracy change:   "
        f"{accuracy_improvement:+.3f} pp"
    )

    print()

    print(
        f"Baseline Macro F1: "
        f"{baseline_f1:.4%}"
    )

    print(
        f"Best Macro F1:     "
        f"{best['macro_f1']:.4%}"
    )

    print(
        f"Macro F1 change:   "
        f"{f1_improvement:+.3f} pp"
    )

    # ========================================================
    # Save CSV
    # ========================================================
    csv_path = (
        output_dir
        / "sgd_hyperparameter_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "experiment",
                "fold1_accuracy",
                "fold2_accuracy",
                "fold3_accuracy",
                "mean_accuracy",
                "macro_f1",
                "accuracy_std",
                "change_vs_baseline_pp",
                "parameters",
            ]
        )

        for result in results:
            change = (
                result["accuracy"]
                - baseline_accuracy
            ) * 100

            writer.writerow(
                [
                    result["name"],
                    result["fold1"],
                    result["fold2"],
                    result["fold3"],
                    result["accuracy"],
                    result["macro_f1"],
                    result["accuracy_std"],
                    change,
                    result["params"],
                ]
            )

    print()
    print(
        "Saved results:",
        csv_path
    )


if __name__ == "__main__":
    main()