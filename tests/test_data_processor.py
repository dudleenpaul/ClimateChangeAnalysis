import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from src.simple_climate_data_processor import SimpleClimateDataProcessor

class TestSimpleClimateDataProcessor(unittest.TestCase):
    """
    Test cases for the SimpleClimateDataProcessor class.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, 'raw')
        self.processed_dir = os.path.join(self.test_dir, 'processed')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create a sample climate data file
        self.test_file = os.path.join(self.raw_dir, 'test_climate_data.csv')
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Year': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
            'Month': [1, 6, 12, 1, 6, 12, 1, 6, 12],
            'Temperature': [0.5, 0.8, 0.7, 0.6, 0.9, 1.0, 0.7, 1.1, 1.2]
        })
        
        sample_data.to_csv(self.test_file, index=False)
        
        # Initialize processor with test directory
        self.processor = SimpleClimateDataProcessor(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_load_climate_data(self):
        """Test loading climate data."""
        df = self.processor.load_climate_data('test_climate_data.csv')
        
        # Check that data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (9, 3))
        self.assertTrue('Year' in df.columns)
        self.assertTrue('Month' in df.columns)
        self.assertTrue('Temperature' in df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(df['Year']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['Month']))
        self.assertTrue(pd.api.types.is_float_dtype(df['Temperature']))
    
    def test_load_climate_data_no_headers(self):
        """Test loading climate data without headers."""
        # Create a file without headers
        no_header_file = os.path.join(self.raw_dir, 'no_header.csv')
        with open(no_header_file, 'w') as f:
            f.write("2020,1,0.5\n")
            f.write("2020,6,0.8\n")
            f.write("2020,12,0.7\n")
        
        df = self.processor.load_climate_data('no_header.csv')
        
        # Check that data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 3))
        self.assertTrue('Year' in df.columns)
        self.assertTrue('Month' in df.columns)
        self.assertTrue('Temperature' in df.columns)
    
    def test_add_derived_features(self):
        """Test adding derived features."""
        df = self.processor.load_climate_data('test_climate_data.csv')
        df_with_features = self.processor.add_derived_features(df)
        
        # Check that new columns are added
        expected_new_columns = ['date', 'decade', 'season', 'temp_5yr_avg', 
                               'temp_10yr_avg', 'temp_anomaly', 'yearly_change']
        
        for col in expected_new_columns:
            self.assertTrue(col in df_with_features.columns)
        
        # Check that decade is calculated correctly
        self.assertEqual(df_with_features.loc[0, 'decade'], 2020)
        
        # Check seasons
        self.assertEqual(df_with_features.loc[0, 'season'], 'Winter')  # January
        self.assertEqual(df_with_features.loc[1, 'season'], 'Summer')  # June
        self.assertEqual(df_with_features.loc[2, 'season'], 'Winter')  # December
    
    def test_preprocess_pipeline(self):
        """Test full preprocessing pipeline."""
        df = self.processor.preprocess_pipeline('test_climate_data.csv')
        
        # Check that the output is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check that it has all expected columns
        expected_columns = ['Year', 'Month', 'Temperature', 'date', 'decade', 
                           'season', 'temp_5yr_avg', 'temp_10yr_avg', 
                           'temp_anomaly', 'yearly_change']
        
        for col in expected_columns:
            self.assertTrue(col in df.columns)
        
        # Check that the processed file was saved
        processed_file = os.path.join(self.processed_dir, 'processed_test_climate_data.csv')
        self.assertTrue(os.path.exists(processed_file))
        
        # Load the processed file and check it
        processed_df = pd.read_csv(processed_file)
        self.assertEqual(processed_df.shape[0], df.shape[0])

if __name__ == '__main__':
    unittest.main()