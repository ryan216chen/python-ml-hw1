from pathlib import Path
import csv
import json

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# HOG Hyperparameter Grid
# ============================================================

HOG_PARAM_GRID = {
    "orientations": [6, 9, 12],
    "pixels_per_cell": [
        (4, 4),
        (7, 7),
    ],
    "cells_per_block": [
        (2, 2),
        (3, 3),
    ],
}


# 固定，不列入第一輪 tuning
BLOCK_NORM = "L2-Hys"
TRANSFORM_SQRT = False


def extract_one_hog(image, params):
    """
    將單張 28x28 MNIST 圖片轉成 HOG feature。
    """
    image = image.reshape(28, 28)

    return hog(
        image,
        orientations=params["orientations"],
        pixels_per_cell=params["pixels_per_cell"],
        cells_per_block=params["cells_per_block"],
        block_norm=BLOCK_NORM,
        transform_sqrt=TRANSFORM_SQRT,
        feature_vector=True,
        channel_axis=None,
    )


def extract_hog_features(x, params):
    """
    對全部 images 做 HOG。

    每張圖片獨立計算，因此先對全部 training set 做 HOG
    不會造成 validation leakage。
    """
    features = Parallel(
        n_jobs=-1,
        batch_size=256,
    )(
        delayed(extract_one_hog)(
            image,
            params
        )
        for image in x
    )

    return np.asarray(
        features,
        dtype=np.float32
    )


def evaluate_hog(x_hog, y):
    """
    使用固定的 3-fold Stratified CV 評估 HOG features。

    SGDClassifier 不調參，
    因此這一輪只有 HOG hyperparameters 是變因。
    """
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=False
    )

    accuracy_scores = []
    f1_scores = []

    for train_idx, valid_idx in cv.split(
        x_hog,
        y
    ):
        x_train = x_hog[train_idx]
        y_train = y[train_idx]

        x_valid = x_hog[valid_idx]
        y_valid = y[valid_idx]

        model = SGDClassifier(
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

    return (
        np.asarray(accuracy_scores),
        np.asarray(f1_scores),
    )


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
    # Load only training data
    # Test set remains untouched
    # ========================================================

    x = np.load(
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
        "Dataset shape:",
        x.shape
    )

    print(
        "Dataset dtype:",
        x.dtype
    )

    # ========================================================
    # Generate all parameter combinations
    # ========================================================

    configurations = list(
        ParameterGrid(
            HOG_PARAM_GRID
        )
    )

    print()
    print(
        "Number of HOG configurations:",
        len(configurations)
    )

    print()

    results = []

    # 原本那組 HOG
    default_hog_accuracy = 0.9747

    # ========================================================
    # HOG parameter search
    # ========================================================

    for index, params in enumerate(
        configurations,
        start=1
    ):
        print(
            "=" * 70
        )

        print(
            f"Configuration "
            f"{index}/{len(configurations)}"
        )

        print(
            "orientations:",
            params["orientations"]
        )

        print(
            "pixels_per_cell:",
            params["pixels_per_cell"]
        )

        print(
            "cells_per_block:",
            params["cells_per_block"]
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Extract HOG
        # ----------------------------------------------------

        x_hog = extract_hog_features(
            x,
            params
        )

        print(
            "HOG feature shape:",
            x_hog.shape
        )

        # ----------------------------------------------------
        # 3-fold CV
        # ----------------------------------------------------

        accuracy_scores, f1_scores = (
            evaluate_hog(
                x_hog,
                y
            )
        )

        mean_accuracy = (
            accuracy_scores.mean()
        )

        mean_f1 = (
            f1_scores.mean()
        )

        accuracy_std = (
            accuracy_scores.std()
        )

        print()

        print(
            "Fold accuracy:"
        )

        for fold, score in enumerate(
            accuracy_scores,
            start=1
        ):
            print(
                f"  Fold {fold}: "
                f"{score:.4%}"
            )

        print()

        print(
            f"Mean Accuracy: "
            f"{mean_accuracy:.4%}"
        )

        print(
            f"Mean Macro F1: "
            f"{mean_f1:.4%}"
        )

        print(
            f"Accuracy std: "
            f"{accuracy_std:.4%}"
        )

        change = (
            mean_accuracy
            - default_hog_accuracy
        ) * 100

        print(
            f"vs Default HOG: "
            f"{change:+.3f} pp"
        )

        results.append(
            {
                "orientations":
                    params["orientations"],

                "pixels_per_cell":
                    params["pixels_per_cell"],

                "cells_per_block":
                    params["cells_per_block"],

                "fold1":
                    accuracy_scores[0],

                "fold2":
                    accuracy_scores[1],

                "fold3":
                    accuracy_scores[2],

                "mean_accuracy":
                    mean_accuracy,

                "macro_f1":
                    mean_f1,

                "accuracy_std":
                    accuracy_std,
            }
        )

        # 刪掉這組 features，
        # 避免下一組時記憶體持續累積
        del x_hog

        print()

    # ========================================================
    # Ranking
    # ========================================================

    results.sort(
        key=lambda result:
            result["mean_accuracy"],
        reverse=True
    )

    print()
    print(
        "=" * 80
    )

    print(
        "HOG Hyperparameter Ranking"
    )

    print(
        "=" * 80
    )

    for rank, result in enumerate(
        results,
        start=1
    ):
        change = (
            result["mean_accuracy"]
            - default_hog_accuracy
        ) * 100

        print(
            f"{rank:2d}. "
            f"ori={result['orientations']:2d} | "
            f"cell={result['pixels_per_cell']} | "
            f"block={result['cells_per_block']} | "
            f"Acc={result['mean_accuracy']:.4%} | "
            f"F1={result['macro_f1']:.4%} | "
            f"Δ={change:+.3f} pp"
        )

    # ========================================================
    # Best configuration
    # ========================================================

    best = results[0]

    print()
    print(
        "=" * 80
    )

    print(
        "BEST HOG CONFIGURATION"
    )

    print(
        "=" * 80
    )

    print(
        "orientations:",
        best["orientations"]
    )

    print(
        "pixels_per_cell:",
        best["pixels_per_cell"]
    )

    print(
        "cells_per_block:",
        best["cells_per_block"]
    )

    print(
        "block_norm:",
        BLOCK_NORM
    )

    print()

    print(
        f"Accuracy: "
        f"{best['mean_accuracy']:.4%}"
    )

    print(
        f"Macro F1: "
        f"{best['macro_f1']:.4%}"
    )

    print(
        f"Accuracy std: "
        f"{best['accuracy_std']:.4%}"
    )

    # ========================================================
    # Save all results as CSV
    # ========================================================

    csv_path = (
        output_dir
        / "hog_hyperparameter_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "rank",
                "orientations",
                "pixels_per_cell",
                "cells_per_block",
                "fold1_accuracy",
                "fold2_accuracy",
                "fold3_accuracy",
                "mean_accuracy",
                "macro_f1",
                "accuracy_std",
            ]
        )

        for rank, result in enumerate(
            results,
            start=1
        ):
            writer.writerow(
                [
                    rank,
                    result[
                        "orientations"
                    ],
                    result[
                        "pixels_per_cell"
                    ],
                    result[
                        "cells_per_block"
                    ],
                    result[
                        "fold1"
                    ],
                    result[
                        "fold2"
                    ],
                    result[
                        "fold3"
                    ],
                    result[
                        "mean_accuracy"
                    ],
                    result[
                        "macro_f1"
                    ],
                    result[
                        "accuracy_std"
                    ],
                ]
            )

    print()
    print(
        "Saved results:",
        csv_path
    )

    # ========================================================
    # Save best configuration as JSON
    # ========================================================

    best_config_path = (
        output_dir
        / "best_hog_config.json"
    )

    best_config = {
        "orientations":
            best["orientations"],

        "pixels_per_cell":
            list(
                best[
                    "pixels_per_cell"
                ]
            ),

        "cells_per_block":
            list(
                best[
                    "cells_per_block"
                ]
            ),

        "block_norm":
            BLOCK_NORM,

        "transform_sqrt":
            TRANSFORM_SQRT,

        "accuracy":
            best[
                "mean_accuracy"
            ],

        "macro_f1":
            best[
                "macro_f1"
            ],
    }

    with open(
        best_config_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            best_config,
            file,
            indent=4
        )

    print(
        "Saved best config:",
        best_config_path
    )

    # ========================================================
    # Rebuild and save ONLY the best HOG features
    #
    # 之後 Feature Fusion / SGD tuning 就可以直接載這個，
    # 不需要重新計算 HOG。
    # ========================================================

    print()
    print(
        "Extracting best HOG features "
        "for future experiments..."
    )

    best_params = {
        "orientations":
            best["orientations"],

        "pixels_per_cell":
            best["pixels_per_cell"],

        "cells_per_block":
            best["cells_per_block"],
    }

    x_best_hog = (
        extract_hog_features(
            x,
            best_params
        )
    )

    best_feature_path = (
        output_dir
        / "mnist_train_best_hog.npy"
    )

    np.save(
        best_feature_path,
        x_best_hog
    )

    print(
        "Saved best HOG features:",
        best_feature_path
    )


if __name__ == "__main__":
    main()