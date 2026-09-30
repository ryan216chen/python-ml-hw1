from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Settings
# ============================================================

# False:
#   Tuned HOG + Normalized Raw Pixel
#
# True:
#   Tuned HOG + Normalized Raw Pixel + Averaged SGD
#
# 這一輪先設 False，單獨測 Feature Fusion。
USE_AVERAGE = True 

# 每次處理多少張圖片建立 fusion feature
CHUNK_SIZE = 2000


def build_fusion_features(
    x_raw,
    x_hog,
    output_path
):
    """
    Feature Fusion:

    Best HOG features
        +
    normalized raw pixels (0~255 -> 0~1)

    使用 open_memmap 分批建立，避免一次占用大量 RAM。
    """

    num_samples = len(x_raw)

    hog_dim = x_hog.shape[1]
    raw_dim = x_raw.shape[1]

    fusion_dim = hog_dim + raw_dim

    print()
    print("Building fusion features...")
    print("HOG dimension:", hog_dim)
    print("Raw pixel dimension:", raw_dim)
    print("Fusion dimension:", fusion_dim)

    # 建立 .npy memmap 檔案
    x_fusion = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            num_samples,
            fusion_dim
        )
    )

    for start in range(
        0,
        num_samples,
        CHUNK_SIZE
    ):
        end = min(
            start + CHUNK_SIZE,
            num_samples
        )

        # ----------------------------------------
        # HOG features
        # ----------------------------------------
        hog_chunk = np.asarray(
            x_hog[start:end],
            dtype=np.float32
        )

        # ----------------------------------------
        # Raw pixel normalization
        # 0~255 -> 0~1
        # ----------------------------------------
        raw_chunk = np.asarray(
            x_raw[start:end],
            dtype=np.float32
        )

        raw_chunk /= 255.0

        # ----------------------------------------
        # Concatenate features
        # ----------------------------------------
        x_fusion[
            start:end,
            :hog_dim
        ] = hog_chunk

        x_fusion[
            start:end,
            hog_dim:
        ] = raw_chunk

        print(
            f"\rProcessed: "
            f"{end}/{num_samples}",
            end=""
        )

    print()
    print(
        "Saved fusion features:",
        output_path
    )

    # 確保寫入硬碟
    x_fusion.flush()

    return x_fusion


def main():
    data_dir = (
        PROJECT_ROOT
        / "data"
        / "mnist"
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    # ========================================================
    # Load original MNIST
    # ========================================================

    x_raw = np.load(
        data_dir / "x.npy",
        mmap_mode="r"
    )[:60000]

    y = np.asarray(
        np.load(
            data_dir / "y.npy",
            mmap_mode="r"
        )[:60000]
    )

    print(
        "Raw image shape:",
        x_raw.shape
    )

    print(
        "Raw image dtype:",
        x_raw.dtype
    )

    # ========================================================
    # Load best tuned HOG features
    #
    # Best HOG:
    # orientations      = 12
    # pixels_per_cell   = (4, 4)
    # cells_per_block   = (3, 3)
    # block_norm        = L2-Hys
    #
    # Accuracy = 97.6733%
    # ========================================================

    hog_path = (
        output_dir
        / "mnist_train_best_hog.npy"
    )

    if not hog_path.exists():
        raise FileNotFoundError(
            f"找不到 Best HOG features:\n"
            f"{hog_path}\n"
            "請先執行 HOG hyperparameter tuning。"
        )

    x_hog = np.load(
        hog_path,
        mmap_mode="r"
    )

    print(
        "Best HOG shape:",
        x_hog.shape
    )

    print(
        "Best HOG dtype:",
        x_hog.dtype
    )

    # ========================================================
    # Build / Load Feature Fusion
    # ========================================================

    fusion_path = (
        output_dir
        / "mnist_train_hog_raw_fusion.npy"
    )

    if fusion_path.exists():
        print()
        print(
            "Loading existing fusion features:",
            fusion_path
        )

        x_fusion = np.load(
            fusion_path,
            mmap_mode="r"
        )

    else:
        build_fusion_features(
            x_raw,
            x_hog,
            fusion_path
        )

        # 建完後重新以 read-only mmap 載入
        x_fusion = np.load(
            fusion_path,
            mmap_mode="r"
        )

    print()
    print(
        "Fusion feature shape:",
        x_fusion.shape
    )

    print(
        "Fusion feature dtype:",
        x_fusion.dtype
    )

    # 預期：
    #
    # 2700 HOG
    # +
    # 784 raw pixel
    # =
    # 3484 features

    # ========================================================
    # 3-fold CV
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

    if USE_AVERAGE:
        print(
            "Experiment: "
            "Tuned HOG + Raw Pixel Fusion "
            "+ Average"
        )
    else:
        print(
            "Experiment: "
            "Tuned HOG + Raw Pixel Fusion"
        )

    print()

    print(
        "Fold | Fusion Accuracy | Macro F1"
    )

    # ========================================================
    # Cross Validation
    # ========================================================

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(
            x_fusion,
            y
        ),
        start=1
    ):
        # ----------------------------------------
        # SGDClassifier
        # ----------------------------------------

        model = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,

            # 這輪 Feature Fusion 單獨測試時 False
            average=USE_AVERAGE,

            max_iter=1000,
            tol=1e-3,

            random_state=42,
            n_jobs=-1
        )

        # ----------------------------------------
        # Train
        #
        # 注意：
        # 使用 indexing 時會建立該 fold 的 array，
        # 但只一次處理一個 fold，
        # 不會同時複製三份。
        # ----------------------------------------

        x_train = x_fusion[
            train_idx
        ]

        y_train = y[
            train_idx
        ]

        model.fit(
            x_train,
            y_train
        )

        # training 用完即可刪掉 reference
        del x_train

        # ----------------------------------------
        # Validation
        # ----------------------------------------

        x_valid = x_fusion[
            valid_idx
        ]

        y_valid = y[
            valid_idx
        ]

        predictions = model.predict(
            x_valid
        )

        del x_valid

        # ----------------------------------------
        # Metrics
        # ----------------------------------------

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

        oof_predictions[
            valid_idx
        ] = predictions

        print(
            f"{fold:4d} | "
            f"{accuracy:15.4%} | "
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

    mean_accuracy = (
        accuracy_scores.mean()
    )

    mean_f1 = (
        f1_scores.mean()
    )

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
    print(
        "=" * 60
    )

    print(
        "Feature Fusion Results"
    )

    print(
        "=" * 60
    )

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
    # Compare with Best HOG
    # ========================================================

    best_hog_accuracy = 0.976733
    best_hog_f1 = 0.976594

    accuracy_change = (
        oof_accuracy
        - best_hog_accuracy
    ) * 100

    f1_change = (
        oof_macro_f1
        - best_hog_f1
    ) * 100

    print()
    print(
        "=" * 60
    )

    print(
        "Compare with Best Tuned HOG"
    )

    print(
        "=" * 60
    )

    print(
        f"Best HOG Accuracy: "
        f"{best_hog_accuracy:.4%}"
    )

    print(
        f"Fusion Accuracy:   "
        f"{oof_accuracy:.4%}"
    )

    print(
        f"Accuracy change:   "
        f"{accuracy_change:+.3f} pp"
    )

    print()

    print(
        f"Best HOG Macro F1: "
        f"{best_hog_f1:.4%}"
    )

    print(
        f"Fusion Macro F1:   "
        f"{oof_macro_f1:.4%}"
    )

    print(
        f"Macro F1 change:   "
        f"{f1_change:+.3f} pp"
    )

    # ========================================================
    # Save OOF predictions
    # ========================================================

    if USE_AVERAGE:
        prediction_path = (
            output_dir
            / "hog_fusion_average_oof_predictions.npy"
        )

    else:
        prediction_path = (
            output_dir
            / "hog_fusion_oof_predictions.npy"
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