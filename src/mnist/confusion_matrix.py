import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    data_dir = PROJECT_ROOT / "data" / "mnist"

    x = np.load(data_dir / "x.npy")
    y = np.load(data_dir / "y.npy")

    x_train = x[:60000]
    y_train = y[:60000]

    model = SGDClassifier(
        random_state=42
    )

    y_pred = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=3,
        n_jobs=3,
        verbose=1
    )

    cm = confusion_matrix(
        y_train,
        y_pred 
    )

    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.arange(10)
    )


    disp.plot()
    plt.title("MNIST Baseline Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()