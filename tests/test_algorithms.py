import unittest
import numpy as np
from src.algorithms import CustomTemperaturePredictor, custom_clustering, detect_anomalies

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2000, 1], [2001, 2], [2002, 3]])
        self.y = np.array([0.1, 0.2, 0.15])

    def test_predictor_fit_and_predict(self):
        model = CustomTemperaturePredictor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_clustering_output_shape(self):
        labels = custom_clustering(self.X, n_clusters=2)
        self.assertEqual(len(labels), len(self.X))

    def test_anomaly_detection(self):
        time_series = np.array([0.1, 0.15, 2.5, 0.12, 0.14, 0.13, 2.6])  # Spikes
        anomalies = detect_anomalies(time_series, window_size=3, threshold=1.0)
        self.assertEqual(len(anomalies), len(time_series))
        self.assertTrue(any(anomalies))

if __name__ == '__main__':
    unittest.main()
