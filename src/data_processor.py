import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Module for loading, cleaning, and processing climate data.
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the DataProcessor with path to data directory.
        
        Args:
            data_dir: Path to directory containing climate data files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.data = None
    
 
 def load_data():
    """Load and prepare the climate data."""
    global data
    try:
        # Check if data file exists
        data_path = os.path.join('data', 'raw', DATA_FILE)
        print(f"Attempting to load data from: {os.path.abspath(data_path)}")
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            return False
        
        # Load and process the data
        print("Loading data...")
        data = data_processor.preprocess_pipeline(DATA_FILE)
        print(f"Data loaded successfully! Shape: {data.shape}, Columns: {data.columns}")
        
        return True
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
@app.route('/test-image')
def test_image():
    try:
        # Create a simple test plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
        ax.set_title('Test Plot')
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Return as HTML for direct viewing
        return f'<img src="data:image/png;base64,{img_data}" />'
    except Exception as e:
        return f"Error generating plot: {str(e)}"
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and removing outliers.
        
        Args:
            data: DataFrame to clean. If None, uses self.data
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Cleaning data...")
        
        # Handle missing values
        data = data.dropna(subset=['temperature', 'year'])  # Drop rows with missing critical values
        
        # Fill missing values for less critical columns
        for col in data.columns:
            if col not in ['temperature', 'year'] and data[col].isna().any():
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    data[col] = data[col].fillna(data[col].mode()[0])
        
        # Remove outliers (simple IQR method)
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
        
        self.data = data
        logger.info(f"Data cleaned. New shape: {data.shape}")
        return data
    
    def normalize_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Normalize numerical columns in the dataset.
        
        Args:
            data: DataFrame to normalize. If None, uses self.data
            
        Returns:
            Normalized DataFrame
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Normalizing data...")
        
        # Create a copy to avoid modifying the original
        normalized_data = data.copy()
        
        # Normalize numerical columns
        for col in data.select_dtypes(include=[np.number]).columns:
            # Skip year column
            if col == 'year':
                continue
                
            min_val = data[col].min()
            max_val = data[col].max()
            
            if max_val > min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
            
        self.data = normalized_data
        logger.info("Data normalized")
        return normalized_data
    
    def preprocess_pipeline(self, filename: str) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline: load, clean, and normalize.
        
        Args:
            filename: Name of data file to process
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting preprocessing pipeline for {filename}")
        
        # Load data
        data = self.load_data(filename)
        
        # Clean data
        data = self.clean_data(data)
        
        # Normalize data
        data = self.normalize_data(data)
        
        # Save processed data
        processed_path = os.path.join(self.processed_dir, f"processed_{filename}")
        if processed_path.endswith('.csv'):
            data.to_csv(processed_path, index=False)
        else:
            data.to_json(processed_path, orient='records')
            
        logger.info(f"Preprocessing complete. Processed data saved to {processed_path}")
        return data
        
    def get_time_series(self, column: str, region: Optional[str] = None) -> pd.DataFrame:
        """
        Extract time series data for a specific column and optional region.
        
        Args:
            column: Column name to extract
            region: Optional region filter
            
        Returns:
            DataFrame with time series data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'year' not in self.data.columns:
            raise ValueError("Data doesn't contain 'year' column required for time series.")
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        if region and 'region' in self.data.columns:
            time_series = self.data[self.data['region'] == region][['year', column]]
        else:
            time_series = self.data[['year', column]]
        
        return time_series.sort_values(by='year')