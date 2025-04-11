import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple

class CustomTemperaturePredictor(BaseEstimator, RegressorMixin):
    """
    A machine learning model that impliments linear regression to predict the temperature

    It inherits BaseEstimator and RegressorMixin from the sklearn API
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the CustomTemperaturePredictor class

        Args:
            learning_rate (float): the learning rate of the model
            n_iterations (int): the number of iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTemperaturePredictor':
        """
        Train a simple linear regression model using gradient descent.

        Args:
            X (np.ndarray): the x-axis of the data
            y (np.ndarray): the y-axis of the data
        Returns:
            CustomTemperaturePredictor
        """
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
