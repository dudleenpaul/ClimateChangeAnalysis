import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple
from sklearn.cluster import KMeans

class CustomTemperaturePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTemperaturePredictor':
        """Train a simple linear regression model using gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
def custom_clustering(data: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Cluster the data into n_clusters using KMeans.
    Returns an array of cluster labels.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels


def detect_anomalies(time_series: np.ndarray, window_size: int = 12, threshold: float = 2.0) -> np.ndarray:
    """
    Detect anomalies in a time series based on a rolling window.
    An anomaly is defined as a point that deviates from the rolling mean by more than `threshold` * std dev.
    Returns a boolean array where True indicates an anomaly.
    """
    anomalies = []
    for i in range(len(time_series)):
        if i < window_size:
            anomalies.append(False)
            continue
        window = time_series[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)
        deviation = abs(time_series[i] - mean)
        is_anomaly = deviation > (threshold * std)
        anomalies.append(is_anomaly)
    return np.array(anomalies)
