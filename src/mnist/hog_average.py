from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"
    output_dir = PROJECT_ROOT / "outputs"

    # ========================================================
    # Load labels
    # ========================================================
    y = np.asarray(
        np.load(
            data_dir / "y.npy",
            mmap_mode="r"
        )[:60000]
    )

    # ========================================================
    # Load best HOG features
    #
    # Best HOG:
    # orientations = 12
    # pixels_per_cell = (4, 4)
    # cells_per_block = (3, 3)
    # ========================================================
    hog_path = (
        output_dir
        / "mnist_train_best_hog.npy"
    )

    if not hog_path.exists():
        raise FileNotFoundError(
            f"找不到 HOG features: {hog_path}\n"
            "請先執行 HOG hyperparameter tuning 程式。"
        )

    x_hog = np.load(
        hog_path,
        mmap_mode="r"
    )

    print("HOG feature shape:", x_hog.shape)
    print("HOG dtype:", x_hog.dtype)

    # ========================================================
    # Same 3-fold CV
    # ========================================================
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=False
    )

    accuracy_scores = []
    f1_scores = []

    oof_predictions = np.empty(
        len(y),
        dtype=y.dtype
    )

    print()
    print("Fold | HOG + Average Accuracy | Macro F1")

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x_hog, y),
        start=1
    ):
        x_train = x_hog[train_idx]
        y_train = y[train_idx]

        x_valid = x_hog[valid_idx]
        y_valid = y[valid_idx]

        # ====================================================
        # HOG + Averaged SGD
        # ====================================================
        model = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            average=True,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            x_train,
            y_train
        )

        predictions = model.predict(
            x_valid
        )

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        macro_f1 = f1_score(
            y_valid,
            predictions,
            average="macro"
        )

        accuracy_scores.append(
            accuracy
        )

        f1_scores.append(
            macro_f1
        )

        oof_predictions[valid_idx] = predictions

        print(
            f"{fold:4d} | "
            f"{accuracy:22.4%} | "
            f"{macro_f1:8.4%}"
        )

    # ========================================================
    # Results
    # ========================================================
    accuracy_scores = np.asarray(
        accuracy_scores
    )

    f1_scores = np.asarray(
        f1_scores
    )

    mean_accuracy = accuracy_scores.mean()
    mean_f1 = f1_scores.mean()

    oof_accuracy = accuracy_score(
        y,
        oof_predictions
    )

    oof_macro_f1 = f1_score(
        y,
        oof_predictions,
        average="macro"
    )

    print()
    print("===== HOG + Average Results =====")

    print(
        f"Mean Accuracy: "
        f"{mean_accuracy:.4%}"
    )

    print(
        f"Mean Macro F1: "
        f"{mean_f1:.4%}"
    )

    print()
    print(
        f"Overall OOF Accuracy: "
        f"{oof_accuracy:.4%}"
    )

    print(
        f"Overall OOF Macro F1: "
        f"{oof_macro_f1:.4%}"
    )

    # ========================================================
    # Compare with tuned HOG baseline
    # ========================================================
    hog_accuracy = 0.976733
    hog_f1 = 0.976594

    accuracy_change = (
        oof_accuracy
        - hog_accuracy
    ) * 100

    f1_change = (
        oof_macro_f1
        - hog_f1
    ) * 100

    print()
    print("===== Compare with Best HOG =====")

    print(
        f"Best HOG Accuracy:      "
        f"{hog_accuracy:.4%}"
    )

    print(
        f"HOG + Average Accuracy: "
        f"{oof_accuracy:.4%}"
    )

    print(
        f"Accuracy change:        "
        f"{accuracy_change:+.3f} pp"
    )

    print()

    print(
        f"Best HOG Macro F1:      "
        f"{hog_f1:.4%}"
    )

    print(
        f"HOG + Average Macro F1: "
        f"{oof_macro_f1:.4%}"
    )

    print(
        f"Macro F1 change:        "
        f"{f1_change:+.3f} pp"
    )

    # ========================================================
    # Save prediction
    # ========================================================
    prediction_path = (
        output_dir
        / "hog_average_oof_predictions.npy"
    )

    np.save(
        prediction_path,
        oof_predictions
    )

    print()
    print(
        "Saved predictions:",
        prediction_path
    )


if __name__ == "__main__":
    main()