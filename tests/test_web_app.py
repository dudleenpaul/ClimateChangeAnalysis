import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from src.visualization import ClimateVisualizer

class TestClimateVisualizer(unittest.TestCase):
    """
    Test cases for the ClimateVisualizer class.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for visualizations
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data for testing
        np.random.seed(42)
        years = np.arange(1900, 2023)
        temps = 0.01 * (years - 1900) + np.random.normal(0, 0.1, len(years))
        
        self.sample_data = pd.DataFrame({
            'Year': years,
            'Temperature': temps
        })
        
        # Add season data for testing
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        self.season_data = pd.DataFrame({
            'season': seasons,
            'Temperature': [0.5, 0.8, 1.2, 0.7]
        })
        
        # Initialize visualizer
        self.visualizer = ClimateVisualizer(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        fig = self.visualizer.plot_time_series(
            self.sample_data,
            x_col='Year',
            y_col='Temperature',
            title='Test Time Series',
            save_path='test_time_series.png'
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_time_series.png')))
    
    def test_plot_time_series_with_grouping(self):
        """Test time series plotting with groups."""
        # Create data with groups
        grouped_data = pd.DataFrame({
            'Year': np.repeat(np.arange(2000, 2010), 3),
            'Temperature': np.random.normal(1.0, 0.2, 30),
            'region': np.tile(['A', 'B', 'C'], 10)
        })
        
        fig = self.visualizer.plot_time_series(
            grouped_data,
            x_col='Year',
            y_col='Temperature',
            title='Test Grouped Time Series',
            group_col='region',
            save_path='test_grouped_time_series.png'
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_grouped_time_series.png')))
    
    def test_plot_heatmap(self):
        """Test heatmap plotting."""
        # Create data for heatmap
        heatmap_data = pd.DataFrame({
            'Year': np.repeat(np.arange(2000, 2010), 4),
            'season': np.tile(['Winter', 'Spring', 'Summer', 'Fall'], 10),
            'Temperature': np.random.normal(1.0, 0.5, 40)
        })
        
        fig = self.visualizer.plot_heatmap(
            heatmap_data,
            value_col='Temperature',
            x_col='Year',
            y_col='season',
            title='Test Heatmap',
            save_path='test_heatmap.png'
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_heatmap.png')))
    
    def test_plot_regional_comparison(self):
        """Test regional comparison plotting."""
        # Create data with regions
        regional_data = pd.DataFrame({
            'region': ['North America', 'Europe', 'Asia', 'Australia'],
            'Temperature': [0.8, 1.2, 1.5, 0.9]
        })
        
        fig = self.visualizer.plot_regional_comparison(
            regional_data,
            value_col='Temperature',
            regions=regional_data['region'].tolist(),
            title='Test Regional Comparison',
            save_path='test_regional.png'
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_regional.png')))
    
    def test_plot_prediction_comparison(self):
        """Test prediction comparison plotting."""
        # Create historical data
        historical_data = self.sample_data[:100].copy()
        
        # Create prediction data
        predicted_data = pd.DataFrame({
            'year': np.arange(2000, 2010),
            'predicted_Temperature': np.linspace(1.0, 1.5, 10)
        })
        
        fig = self.visualizer.plot_prediction_comparison(
            historical_data,
            predicted_data,
            x_col='Year',
            actual_col='Temperature',
            predicted_col='predicted_Temperature',
            title='Test Prediction Comparison',
            save_path='test_prediction.png'
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_prediction.png')))
    
    def test_plot_anomalies(self):
        """Test anomaly plotting."""
        # Create data with anomalies
        anomaly_data = self.sample_data.copy()
        anomaly_data['is_anomaly'] = False
        anomaly_data.loc[[20, 50, 80], 'is_anomaly'] = True
        
        fig = self.visualizer.plot_anomalies(
            anomaly_data,
            x_col='Year',
            y_col='Temperature',
            title='Test Anomalies',
            save_path='test_anomalies.png'
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_anomalies.png')))

if __name__ == '__main__':
    unittest.main()