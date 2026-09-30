from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def shift_one_pixel(x, rng):
    """
    將每張 28x28 MNIST 圖片隨機向
    上、下、左、右其中一個方向平移 1 pixel。
    """
    original = x.reshape(-1, 28, 28)
    shifted = np.zeros_like(original)

    # 0: up
    # 1: down
    # 2: left
    # 3: right
    directions = rng.integers(0, 4, size=len(x))

    up = np.flatnonzero(directions == 0)
    down = np.flatnonzero(directions == 1)
    left = np.flatnonzero(directions == 2)
    right = np.flatnonzero(directions == 3)

    shifted[up, :-1, :] = original[up, 1:, :]
    shifted[down, 1:, :] = original[down, :-1, :]
    shifted[left, :, :-1] = original[left, :, 1:]
    shifted[right, :, 1:] = original[right, :, :-1]

    return shifted.reshape(-1, 784)


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"

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

    # 儲存每一張圖片的 out-of-fold prediction
    # 之後若要算其他 metric / confusion matrix，
    # 不需要重新訓練模型
    oof_predictions = np.empty(
        len(y),
        dtype=np.asarray(y).dtype
    )

    print()
    print(
        "Fold | Augmented accuracy | Macro F1"
    )

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x, y),
        start=1
    ):
        # -----------------------------
        # Train / Validation split
        # -----------------------------
        x_train = np.asarray(x[train_idx])
        y_train = np.asarray(y[train_idx])

        x_valid = np.asarray(x[valid_idx])
        y_valid = np.asarray(y[valid_idx])

        # -----------------------------
        # Data Augmentation
        # -----------------------------
        x_shifted = shift_one_pixel(
            x_train,
            np.random.default_rng(42 + fold)
        )

        # 原圖 + 平移後圖片
        x_augmented = np.concatenate(
            (
                x_train,
                x_shifted
            ),
            axis=0
        )

        y_augmented = np.concatenate(
            (
                y_train,
                y_train
            ),
            axis=0
        )

        # -----------------------------
        # Train
        # -----------------------------
        model = SGDClassifier(
            random_state=42,

            # MNIST 是 multiclass，
            # sklearn 可平行處理 one-vs-rest
            n_jobs=-1
        )

        model.fit(
            x_augmented,
            y_augmented
        )

        # -----------------------------
        # Predict 一次
        # Accuracy / F1 共用
        # -----------------------------
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

        # 存回原本 validation index
        oof_predictions[valid_idx] = predictions

        print(
            f"{fold:4d} | "
            f"{accuracy:18.4%} | "
            f"{macro_f1:8.4%}"
        )

    # -----------------------------
    # Fold average
    # -----------------------------
    accuracy_scores = np.array(
        accuracy_scores
    )

    f1_scores = np.array(
        f1_scores
    )

    print()
    print(
        f"Mean augmented accuracy: "
        f"{accuracy_scores.mean():.4%}"
    )

    print(
        f"Mean augmented Macro F1: "
        f"{f1_scores.mean():.4%}"
    )

    # -----------------------------
    # Overall OOF metrics
    # 等同 cross_val_predict 的概念
    # -----------------------------
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

    # -----------------------------
    # Save predictions
    # -----------------------------
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "augmentation_oof_predictions.npy"
    )

    np.save(
        output_path,
        oof_predictions
    )

    print()
    print(
        f"Saved predictions: {output_path}"
    )


if __name__ == "__main__":
    main()