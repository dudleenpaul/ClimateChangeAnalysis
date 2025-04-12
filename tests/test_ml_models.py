import unittest
import pandas as pd
import numpy as np
from src.ml_models import ClimateMLModels

class TestClimateMLModels(unittest.TestCase):
    """
    Test cases for the ClimateMLModels class.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create sample data for testing
        np.random.seed(42)
        years = np.arange(1900, 2023)
        temps = 0.01 * (years - 1900) + np.random.normal(0, 0.1, len(years))
        
        self.sample_data = pd.DataFrame({
            'Year': years,
            'Temperature': temps
        })
        
        # Initialize models
        self.ml_models = ClimateMLModels()
    
    def test_train_prediction_model(self):
        """Test training a prediction model."""
        metrics = self.ml_models.train_prediction_model(
            self.sample_data,
            target='Temperature',
            features=['Year'],
            model_type='linear'
        )
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertTrue('r2' in metrics)
        self.assertTrue('mse' in metrics)
        
        # Check that model was stored
        self.assertIsNotNone(self.ml_models.prediction_model)
        self.assertEqual(self.ml_models.prediction_features, ['Year'])
        self.assertEqual(self.ml_models.prediction_target, 'Temperature')
        
        # Check with random forest model
        metrics_rf = self.ml_models.train_prediction_model(
            self.sample_data,
            target='Temperature',
            features=['Year'],
            model_type='random_forest'
        )
        
        self.assertIsInstance(metrics_rf, dict)
    
    def test_predict_future(self):
        """Test predicting future values."""
        # First train a model
        self.ml_models.train_prediction_model(
            self.sample_data,
            target='Temperature',
            features=['Year']
        )
        
        # Create future data
        future_data = pd.DataFrame({'Year': np.arange(2023, 2033)})
        
        # Make predictions
        predictions = self.ml_models.predict_future(future_data, years_ahead=10)
        
        # Check predictions
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), 10)
        self.assertTrue('predicted_Temperature' in predictions.columns)
        self.assertTrue('year' in predictions.columns)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # Add some anomalies to the data
        data_with_anomalies = self.sample_data.copy()
        # Add extreme values as anomalies
        anomaly_indices = [20, 50, 80, 110]
        for idx in anomaly_indices:
            data_with_anomalies.loc[idx, 'Temperature'] += 1.0
        
        # Detect anomalies
        result = self.ml_models.detect_anomalies(
            data_with_anomalies,
            features=['Temperature'],
            contamination=0.05
        )
        
        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('is_anomaly' in result.columns)
        
        # Check that some anomalies were detected
        self.assertTrue(result['is_anomaly'].sum() > 0)
    
    def test_cluster_regions(self):
        """Test region clustering."""
        # Create sample data with multiple regions
        np.random.seed(42)
        regions = ['A', 'B', 'C']
        years = np.repeat(np.arange(2000, 2010), len(regions))
        region_values = np.tile(regions, 10)
        
        # Different temperature patterns for each region
        temps = np.zeros(len(years))
        for i, region in enumerate(region_values):
            if region == 'A':
                temps[i] = 0.5 + 0.1 * (years[i] - 2000) + np.random.normal(0, 0.05)
            elif region == 'B':
                temps[i] = 1.0 + 0.05 * (years[i] - 2000) + np.random.normal(0, 0.05)
            else:  # C
                temps[i] = 1.5 - 0.02 * (years[i] - 2000) + np.random.normal(0, 0.05)
        
        # Create precipitation data
        precip = np.zeros(len(years))
        for i, region in enumerate(region_values):
            if region == 'A':
                precip[i] = 100 + np.random.normal(0, 5)
            elif region == 'B':
                precip[i] = 150 + np.random.normal(0, 8)
            else:  # C
                precip[i] = 80 + np.random.normal(0, 3)
        
        multi_region_data = pd.DataFrame({
            'Year': years,
            'region': region_values,
            'Temperature': temps,
            'Precipitation': precip
        })
        
        # Test clustering
        result = self.ml_models.cluster_regions(
            multi_region_data,
            features=['Temperature', 'Precipitation'],
            n_clusters=3
        )
        
        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('cluster' in result.columns)
        
        # Check that we have the right number of clusters
        self.assertEqual(len(result['cluster'].unique()), 3)

if __name__ == '__main__':
    unittest.main()