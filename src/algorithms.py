# src/algorithms.py
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple

class CustomTemperaturePredictor(BaseEstimator, RegressorMixin):
    """
    A custom machine learning model that predicts temperature

    This class inherits the sklearn.base modules BaseEstimator and RegressorMixin

    Attributes:
        learning_rate (float): the rate at which the model learns
        n_iterations (int): the number of iterations the model will go through
        weights (<type>): <description>
        bias (<type>): <description>
    Methods:
        fit(self, X, y): Train a simple linear regression model using gradient descent
            retuns CustomTemperaturePredictor
        predict(self, X): <description of method>
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """Initalize the CustomeTemperaturePredictor object. """        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTemperaturePredictor':
        """
        Train a simple linear regression model using gradient descent.
        
        Args:
            X (np.ndarray): array of points for the x-axis
            y (np.ndarray): array of points for the y-axis
        Returns:
            this CustomTemperaturePredictor object 
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            self.weights -= self.learning_rate * (1/n_samples) * np.dot(X.T, error)
            self.bias -= self.learning_rate * (1/n_samples) * np.sum(error)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


def custom_clustering(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Custom clustering algorithm using k-means.

    Args:
        data (np.ndarray): the data to be sorted into clusters as a NumPy array
    Kwargs:
        n_clusters (int): number of clusters to form
    Returns:
        an array of cluster labels
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)


def detect_anomalies(time_series: np.ndarray, window_size: int = 10, threshold: float = 2.0) -> np.ndarray:
    """
    Detect anomalies in time series data using rolling z-score.
    An anomaly is defined as a point that deviates from the rolling mean by more than 'threshold' * std dev.
    
    Args:
        time_series (np.ndarray): the input time series as a NumPy array
    Kwargs:
        window_size (int): number of previous time steps used to compute the rolling mean and standard deviation
        threshold (float): how far a point must be to be considered an anomaly
    Returns:
        a boolean array where True indicates an anomaly and False indicates no anomaly
    """
    rolling_mean = np.convolve(time_series, np.ones(window_size)/window_size, mode='valid')
    z_scores = (time_series[window_size-1:] - rolling_mean) / np.std(time_series[window_size-1:])
    anomalies = np.abs(z_scores) > threshold
    padded = np.concatenate((np.zeros(window_size-1, dtype=bool), anomalies))
    return padded
