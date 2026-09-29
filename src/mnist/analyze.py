import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"

    x = np.load(data_dir / "x.npy")
    y = np.load(data_dir / "y.npy")

    x_train = x[:60000]
    y_train = y[:60000]

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    classes, counts = np.unique(y_train, return_counts=True)

    print("Classes:", classes)

    for cls, count in zip(classes, counts):
        print(f"Class {cls}: {count}")

    
    print("Pixel min:", x_train.min())
    print("Pixel max:", x_train.max())

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for cls, ax in zip(classes, axes.flat):
        index = np.where(y_train == cls)[0][0]

        image = x_train[index].reshape(28, 28)

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {cls}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()