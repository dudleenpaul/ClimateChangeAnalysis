import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from src.simple_climate_data_processor import SimpleClimateDataProcessor
from src.ml_models import ClimateMLModels
from src.visualization import ClimateVisualizer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import seaborn as sns

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the Climate Change Impact Analyzer.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, 'raw')
        self.processed_dir = os.path.join(self.test_dir, 'processed')
        self.viz_dir = os.path.join(self.test_dir, 'visualizations')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Create sample climate data file
        self.test_file = os.path.join(self.raw_dir, 'test_climate_data.csv')
        
        # Create more comprehensive sample data
        np.random.seed(42)
        years = np.repeat(np.arange(1950, 2023), 12)
        months = np.tile(np.arange(1, 13), 73)
        
        # Create temperature data with trend and seasonal patterns
        base_temp = 0.0
        trend = 0.01 * (years - 1950)  # Warming trend
        seasonal = 0.5 * np.sin(2 * np.pi * (months - 6) / 12)  # Seasonal cycle
        noise = np.random.normal(0, 0.2, len(years))  # Random variations
        
        temps = base_temp + trend + seasonal + noise
        
        sample_data = pd.DataFrame({
            'Year': years,
            'Month': months,
            'Temperature': temps
        })
        
        sample_data.to_csv(self.test_file, index=False)
        
        # Initialize components
        self.data_processor = SimpleClimateDataProcessor(data_dir=self.test_dir)
        self.ml_models = ClimateMLModels()
        self.visualizer = ClimateVisualizer(output_dir=self.viz_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_full_pipeline(self):
        """Test the full data processing and analysis pipeline."""
        # Process data
        processed_data = self.data_processor.preprocess_pipeline('test_climate_data.csv')
        
        # Check processed data
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertTrue('Year' in processed_data.columns)
        self.assertTrue('decade' in processed_data.columns)
        self.assertTrue('season' in processed_data.columns)
        
        # Prepare yearly data for prediction
        yearly_data = processed_data.groupby('Year')['Temperature'].mean().reset_index()
        
        # Train prediction model
        metrics = self.ml_models.train_prediction_model(
            yearly_data,
            target='Temperature',
            features=['Year']
        )
        
        # Check model metrics
        self.assertIsInstance(metrics, dict)
        self.assertTrue('r2' in metrics)
        self.assertTrue(metrics['r2'] > 0.5)  # Expect reasonable fit due to trend
        
        # Make future predictions
        last_year = yearly_data['Year'].max()
        future_years = 10
        future_data = pd.DataFrame({'Year': np.arange(last_year + 1, last_year + future_years + 1)})
        
        predictions = self.ml_models.predict_future(future_data, future_years)
        
        # Check predictions
        self.assertEqual(len(predictions), future_years)
        self.assertTrue('predicted_Temperature' in predictions.columns)
        
        # Create visualization
        fig = self.visualizer.plot_prediction_comparison(
            yearly_data,
            predictions,
            x_col='Year',
            actual_col='Temperature',
            predicted_col='predicted_Temperature',
            title='Temperature Predictions',
            save_path='prediction_test.png'
        )
        
        # Check visualization
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(os.path.join(self.viz_dir, 'prediction_test.png')))
        
        # Detect anomalies
        anomaly_data = self.ml_models.detect_anomalies(
            yearly_data,
            features=['Temperature']
        )
        
        # Check anomalies
        self.assertTrue('is_anomaly' in anomaly_data.columns)
        
        # Create anomaly visualization
        fig = self.visualizer.plot_anomalies(
            anomaly_data,
            x_col='Year',
            y_col='Temperature',
            title='Temperature Anomalies',
            save_path='anomaly_test.png'
        )
        
        # Check visualization
        self.assertTrue(os.path.exists(os.path.join(self.viz_dir, 'anomaly_test.png')))
    
    def test_seasonal_analysis(self):
        """Test seasonal analysis pipeline."""
        # Process data
        processed_data = self.data_processor.preprocess_pipeline('test_climate_data.csv')
        
        # Compute seasonal averages
        seasonal_data = processed_data.groupby('season')['Temperature'].mean().reset_index()
        
        # Check seasonal data
        self.assertEqual(len(seasonal_data), 4)  # 4 seasons
        self.assertListEqual(sorted(seasonal_data['season'].tolist()), 
                            sorted(['Winter', 'Spring', 'Summer', 'Fall']))
        
        # Create seasonal visualization
        fig, ax = plt.subplots()
        sns.barplot(x='season', y='Temperature', data=seasonal_data, ax=ax)
        plt.savefig(os.path.join(self.viz_dir, 'seasonal_test.png'))
        plt.close(fig)
        
        # Check visualization was created
        self.assertTrue(os.path.exists(os.path.join(self.viz_dir, 'seasonal_test.png')))
        
        # Compare different decades
        decade_seasonal = processed_data.groupby(['decade', 'season'])['Temperature'].mean().reset_index()
        
        # Check we have data for multiple decades
        self.assertTrue(len(decade_seasonal['decade'].unique()) > 1)
        
        # Create decade comparison visualization for a single season
        summer_data = decade_seasonal[decade_seasonal['season'] == 'Summer']
        fig, ax = plt.subplots()
        sns.lineplot(x='decade', y='Temperature', data=summer_data, ax=ax, marker='o')
        plt.savefig(os.path.join(self.viz_dir, 'summer_trend_test.png'))
        plt.close(fig)
        
        # Check visualization was created
        self.assertTrue(os.path.exists(os.path.join(self.viz_dir, 'summer_trend_test.png')))
        
    def test_decadal_analysis(self):
        """Test decadal analysis pipeline."""
        # Process data
        processed_data = self.data_processor.preprocess_pipeline('test_climate_data.csv')
        
        # Compute decadal averages
        decadal_data = processed_data.groupby('decade')['Temperature'].mean().reset_index()
        
        # Check decadal data
        self.assertTrue(len(decadal_data) >= 7)  # At least 7 decades (1950s-2010s)
        
        # Create decadal visualization
        fig, ax = plt.subplots()
        bars = ax.bar(decadal_data['decade'], decadal_data['Temperature'])
        
        # Add trend line
        x = decadal_data['decade']
        y = decadal_data['Temperature']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8)
        
        plt.savefig(os.path.join(self.viz_dir, 'decadal_test.png'))
        plt.close(fig)
        
        # Check visualization was created
        self.assertTrue(os.path.exists(os.path.join(self.viz_dir, 'decadal_test.png')))
        
        # Check for warming trend (should be positive due to how we generated the data)
        self.assertTrue(z[0] > 0)

if __name__ == '__main__':
    unittest.main()