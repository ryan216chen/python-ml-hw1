import numpy as np 
from pathlib import Path 

from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import cross_val_score 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"

    x = np.load(data_dir / "x.npy")
    y = np.load(data_dir / "y.npy")

    x_train = x[:60000]
    y_train = y[:60000]

    model = SGDClassifier(
        random_state = 42
    )

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    print("Accuracy of each fold:", scores)
    print("Mean accuracy:", scores.mean())

if __name__ == "__main__":
    main()