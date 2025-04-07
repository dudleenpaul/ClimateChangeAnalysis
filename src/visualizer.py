import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true)), y_true, label='Actual', alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature (normalized)')
    plt.title('Predicted vs. Actual Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
