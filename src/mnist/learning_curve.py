from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import learning_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"
    x = np.load(data_dir / "x.npy")
    y = np.load(data_dir / "y.npy")

    # Use only the original 60,000 training images; keep the test set untouched.
    x_train = x[:60000]
    y_train = y[:60000]

    sizes, train_scores, valid_scores = learning_curve(
        SGDClassifier(random_state=42),
        x_train,
        y_train,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        cv=3,
        scoring="accuracy",
        shuffle=True,
        random_state=42,
        n_jobs=3,
        error_score="raise",
    )

    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    print("Training size | Train accuracy | Validation accuracy")
    for size, train_accuracy, valid_accuracy in zip(sizes, train_mean, valid_mean):
        print(f"{size:13d} | {train_accuracy:14.4f} | {valid_accuracy:19.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, "o-", label="Training accuracy")
    ax.plot(sizes, valid_mean, "o-", label="Cross-validation accuracy")
    ax.fill_between(
        sizes,
        valid_mean - valid_scores.std(axis=1),
        valid_mean + valid_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set(
        title="MNIST baseline learning curve (3-fold CV)",
        xlabel="Training images per fold",
        ylabel="Accuracy",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "mnist_baseline_learning_curve.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
