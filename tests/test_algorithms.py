from src.algorithms import (
    CustomTemperaturePredictor,
    custom_clustering,
    detect_anomalies
)
import numpy as np

def test_fit_and_predict():
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([1, 2, 3])
    model = CustomTemperaturePredictor()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    
def test_custom_clustering():
    X = np.random.rand(20, 2)
    labels = custom_clustering(X, n_clusters=3)
    assert len(labels) == len(X)

def test_detect_anomalies():
    ts = np.array([1]*10 + [10] + [1]*10)
    anomalies = detect_anomalies(ts, window_size=5, threshold=2)
    assert anomalies[10] is True

