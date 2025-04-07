from src.algorithms import CustomTemperaturePredictor
import numpy as np

def test_fit_and_predict():
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([1, 2, 3])
    model = CustomTemperaturePredictor()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
