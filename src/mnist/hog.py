from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# HOG parameters
# ============================================================
ORIENTATIONS = 9
PIXELS_PER_CELL = (4, 4)
CELLS_PER_BLOCK = (2, 2)
BLOCK_NORM = "L2-Hys"

# HOG 參數確定後，最後才建議改成 True 看 test
RUN_TEST = False


def extract_hog(image):
    """
    將一張 28x28 MNIST image 轉換成 HOG feature。
    """
    image = image.reshape(28, 28)

    features = hog(
        image,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm=BLOCK_NORM,
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None
    )

    return features


def extract_hog_features(x, name):
    """
    對一批圖片進行 HOG feature extraction。
    每張圖片獨立處理，因此不會產生 data leakage。
    """
    print()
    print(f"Extracting HOG features for {name}...")

    features = Parallel(
        n_jobs=-1,
        verbose=10
    )(
        delayed(extract_hog)(image)
        for image in x
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        f"{name} HOG shape:",
        features.shape
    )

    return features


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"
    output_dir = PROJECT_ROOT / "outputs"

    output_dir.mkdir(exist_ok=True)

    # ========================================================
    # Load dataset
    # ========================================================
    x = np.load(
        data_dir / "x.npy",
        mmap_mode="r"
    )

    y = np.load(
        data_dir / "y.npy",
        mmap_mode="r"
    )

    # 原本 MNIST:
    # 0 ~ 59999     -> train
    # 60000 ~ 69999 -> test
    x_train = x[:60000]
    y_train = np.asarray(y[:60000])

    x_test = x[60000:]
    y_test = np.asarray(y[60000:])

    print("Training images:", x_train.shape)
    print("Test images:", x_test.shape)
    print("Original dtype:", x.dtype)

    print()
    print("HOG settings")
    print("orientations:", ORIENTATIONS)
    print("pixels_per_cell:", PIXELS_PER_CELL)
    print("cells_per_block:", CELLS_PER_BLOCK)
    print("block_norm:", BLOCK_NORM)

    # ========================================================
    # HOG feature extraction
    # ========================================================
    train_hog_path = (
        output_dir
        / "mnist_train_hog_features.npy"
    )

    # 如果之前算過，就直接載入，避免重算 HOG
    if train_hog_path.exists():
        print()
        print(
            "Loading cached training HOG features:",
            train_hog_path
        )

        x_train_hog = np.load(
            train_hog_path
        )

    else:
        x_train_hog = extract_hog_features(
            x_train,
            "training set"
        )

        np.save(
            train_hog_path,
            x_train_hog
        )

        print(
            "Saved training HOG features:",
            train_hog_path
        )

    print()
    print(
        "Raw feature dimension:",
        x_train.shape[1]
    )

    print(
        "HOG feature dimension:",
        x_train_hog.shape[1]
    )

    # ========================================================
    # 3-fold Cross Validation
    # ========================================================
    # 與原本 baseline cv=3 對齊
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=False
    )

    accuracy_scores = []
    f1_scores = []

    # 儲存每一張 training image 的
    # out-of-fold prediction
    oof_predictions = np.empty(
        len(y_train),
        dtype=y_train.dtype
    )

    print()
    print(
        "Fold | HOG accuracy | Macro F1"
    )

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(x_train_hog, y_train),
        start=1
    ):
        # ----------------------------------------
        # Train / Validation
        # ----------------------------------------
        x_fold_train = x_train_hog[train_idx]
        y_fold_train = y_train[train_idx]

        x_fold_valid = x_train_hog[valid_idx]
        y_fold_valid = y_train[valid_idx]

        # ----------------------------------------
        # SGDClassifier
        # ----------------------------------------
        model = SGDClassifier(
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            x_fold_train,
            y_fold_train
        )

        # predict 一次，
        # Accuracy / F1 共用 prediction
        predictions = model.predict(
            x_fold_valid
        )

        accuracy = accuracy_score(
            y_fold_valid,
            predictions
        )

        macro_f1 = f1_score(
            y_fold_valid,
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
            f"{accuracy:12.4%} | "
            f"{macro_f1:8.4%}"
        )

    # ========================================================
    # CV results
    # ========================================================
    accuracy_scores = np.asarray(
        accuracy_scores
    )

    f1_scores = np.asarray(
        f1_scores
    )

    mean_accuracy = accuracy_scores.mean()
    mean_f1 = f1_scores.mean()

    # 所有 60,000 張 OOF prediction 一起計算
    oof_accuracy = accuracy_score(
        y_train,
        oof_predictions
    )

    oof_macro_f1 = f1_score(
        y_train,
        oof_predictions,
        average="macro"
    )

    print()
    print("===== 3-fold CV results =====")

    print(
        f"Mean HOG accuracy: "
        f"{mean_accuracy:.4%}"
    )

    print(
        f"Mean HOG Macro F1: "
        f"{mean_f1:.4%}"
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

    # ========================================================
    # Compare with original baseline
    # ========================================================
    baseline_accuracy = 0.868533
    baseline_f1 = 0.8670

    accuracy_change = (
        oof_accuracy
        - baseline_accuracy
    ) * 100

    f1_change = (
        oof_macro_f1
        - baseline_f1
    ) * 100

    print()
    print("===== Comparison with baseline =====")

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4%}"
    )

    print(
        f"HOG accuracy:      "
        f"{oof_accuracy:.4%}"
    )

    print(
        f"Accuracy change:   "
        f"{accuracy_change:+.2f} pp"
    )

    print()

    print(
        f"Baseline Macro F1: "
        f"{baseline_f1:.4%}"
    )

    print(
        f"HOG Macro F1:      "
        f"{oof_macro_f1:.4%}"
    )

    print(
        f"Macro F1 change:   "
        f"{f1_change:+.2f} pp"
    )

    # ========================================================
    # Save OOF predictions
    # ========================================================
    oof_path = (
        output_dir
        / "hog_oof_predictions.npy"
    )

    np.save(
        oof_path,
        oof_predictions
    )

    print()
    print(
        "Saved OOF predictions:",
        oof_path
    )

    # ========================================================
    # Final Test
    #
    # 建議 HOG parameters 全部選定後才執行。
    # ========================================================
    if RUN_TEST:
        print()
        print("===== Final Test =====")

        test_hog_path = (
            output_dir
            / "mnist_test_hog_features.npy"
        )

        if test_hog_path.exists():
            print(
                "Loading cached test HOG features:",
                test_hog_path
            )

            x_test_hog = np.load(
                test_hog_path
            )

        else:
            x_test_hog = extract_hog_features(
                x_test,
                "test set"
            )

            np.save(
                test_hog_path,
                x_test_hog
            )

        # 用全部 60,000 training data
        # 訓練 final model
        final_model = SGDClassifier(
            random_state=42,
            n_jobs=-1
        )

        final_model.fit(
            x_train_hog,
            y_train
        )

        test_predictions = final_model.predict(
            x_test_hog
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_macro_f1 = f1_score(
            y_test,
            test_predictions,
            average="macro"
        )

        print()
        print(
            f"Test Accuracy: "
            f"{test_accuracy:.4%}"
        )

        print(
            f"Test Macro F1: "
            f"{test_macro_f1:.4%}"
        )

        test_prediction_path = (
            output_dir
            / "hog_test_predictions.npy"
        )

        np.save(
            test_prediction_path,
            test_predictions
        )

        print(
            "Saved test predictions:",
            test_prediction_path
        )


if __name__ == "__main__":
    main()