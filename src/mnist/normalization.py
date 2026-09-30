from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"

    # 只使用原本 60,000 張 training data
    x = np.load(
        data_dir / "x.npy",
        mmap_mode="r"
    )[:60000]

    y = np.load(
        data_dir / "y.npy",
        mmap_mode="r"
    )[:60000]

    print("Dataset dtype:", x.dtype)
    print("Dataset shape:", x.shape)

    # 與 baseline 的 cv=3 對齊
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=False
    )

    accuracy_scores = []
    f1_scores = []

    # 儲存所有 out-of-fold predictions
    oof_predictions = np.empty(
        len(y),
        dtype=np.asarray(y).dtype
    )

    print()
    print("Fold | Normalized accuracy | Macro F1")

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x, y),
        start=1
    ):
        # =========================================
        # Train / Validation split
        # =========================================
        x_train = np.asarray(
            x[train_idx],
            dtype=np.float32
        )

        y_train = np.asarray(
            y[train_idx]
        )

        x_valid = np.asarray(
            x[valid_idx],
            dtype=np.float32
        )

        y_valid = np.asarray(
            y[valid_idx]
        )

        # =========================================
        # Normalization
        # 0 ~ 255 -> 0 ~ 1
        # =========================================
        x_train /= 255.0
        x_valid /= 255.0

        # =========================================
        # Train
        # =========================================
        model = SGDClassifier(
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            x_train,
            y_train
        )

        # =========================================
        # Predict
        # =========================================
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
            f"{accuracy:19.4%} | "
            f"{macro_f1:8.4%}"
        )

    # =========================================
    # Fold mean
    # =========================================
    accuracy_scores = np.array(
        accuracy_scores
    )

    f1_scores = np.array(
        f1_scores
    )

    print()
    print(
        f"Mean normalized accuracy: "
        f"{accuracy_scores.mean():.4%}"
    )

    print(
        f"Mean normalized Macro F1: "
        f"{f1_scores.mean():.4%}"
    )

    # =========================================
    # Overall OOF metrics
    # =========================================
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
    print("Overall OOF results")

    print(
        f"Accuracy: "
        f"{oof_accuracy:.4%}"
    )

    print(
        f"Macro F1: "
        f"{oof_macro_f1:.4%}"
    )

    # =========================================
    # Compare with baseline
    # =========================================
    baseline_accuracy = 0.868533

    change = (
        oof_accuracy
        - baseline_accuracy
    ) * 100

    print()
    print(
        f"Baseline accuracy:   "
        f"{baseline_accuracy:.4%}"
    )

    print(
        f"Normalized accuracy: "
        f"{oof_accuracy:.4%}"
    )

    print(
        f"Change: "
        f"{change:+.2f} percentage points"
    )

    # =========================================
    # Save predictions
    # =========================================
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "normalization_oof_predictions.npy"
    )

    np.save(
        output_path,
        oof_predictions
    )

    print()
    print(
        f"Saved predictions: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()