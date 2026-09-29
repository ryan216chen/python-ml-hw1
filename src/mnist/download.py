from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"
    data_dir.mkdir(parents=True, exist_ok=True)

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False
    )

    x = mnist.data
    y = mnist.target.astype(np.uint8)

    np.save(data_dir / "x.npy", x)
    np.save(data_dir / "y.npy", y)

    print("Download complete")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x dtype:", x.dtype)
    print("y dtype:", y.dtype)


if __name__ == "__main__":
    main()