import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleClimateDataProcessor:
    """
    Module for processing simple climate data format from Berkeley Earth.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the simple climate data processor.
        
        Args:
            data_dir: Directory containing climate data files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_climate_data(self, filename="climate_data.csv"):
        """
        Load climate data in the format Year,Month,Temperature.
        
        Args:
            filename: Name of the climate data file
            
        Returns:
            DataFrame containing climate data
        """
        file_path = os.path.join(self.raw_dir, filename)
        logger.info(f"Loading climate data from {file_path}")
        
        try:
            # Try to load the file with headers
            df = pd.read_csv(file_path)
            
            # If loaded successfully but columns don't have meaningful names, rename them
            if len(df.columns) == 3 and df.columns[0].isdigit():
                logger.info("File appears to have no headers. Renaming columns.")
                df.columns = ['Year', 'Month', 'Temperature']
        except:
            # If that fails, try loading without headers
            logger.info("Trying to load file without headers")
            df = pd.read_csv(file_path, header=None, names=['Year', 'Month', 'Temperature'])
        
        # Convert data types
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').astype('Int64')
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna()
        
        logger.info(f"Loaded climate data with shape {df.shape}")
        return df
    
    def add_derived_features(self, data):
        """
        Add derived features to the dataset.
        
        Args:
            data: DataFrame containing climate data
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Adding derived features to climate data")
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Add date column for easier time series analysis
        df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
        
        # Add decade column for aggregation
        df['decade'] = (df['Year'] // 10) * 10
        
        # Add season column (Northern Hemisphere seasons)
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
        }
        df['season'] = df['Month'].map(season_map)
        
        # Calculate rolling averages
        # Sort by date first
        df = df.sort_values('date')
        
        # 5-year moving average (60 months)
        df['temp_5yr_avg'] = df['Temperature'].rolling(window=60, min_periods=1).mean()
        
        # 10-year moving average (120 months)
        df['temp_10yr_avg'] = df['Temperature'].rolling(window=120, min_periods=1).mean()
        
        # Add temperature anomaly indicators
        temp_mean = df['Temperature'].mean()
        temp_std = df['Temperature'].std()
        
        df['temp_anomaly'] = np.abs(df['Temperature'] - temp_mean) > (2 * temp_std)
        
        # Calculate year-to-year temperature change
        yearly_avg = df.groupby('Year')['Temperature'].mean()
        yearly_change = yearly_avg.diff()
        
        # Map yearly change back to the main dataframe
        year_change_dict = yearly_change.to_dict()
        df['yearly_change'] = df['Year'].map(year_change_dict)
        
        logger.info(f"Derived features added. New shape: {df.shape}")
        return df
    
    def preprocess_pipeline(self, filename="climate_data.csv"):
        """
        Run complete preprocessing pipeline for climate data.
        
        Args:
            filename: Name of the climate data file
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting climate data preprocessing pipeline")
        
        # Load climate data
        data = self.load_climate_data(filename)
        
        if data is None or data.empty:
            logger.warning("No data to process")
            return None
        
        # Add derived features
        data = self.add_derived_features(data)
        
        # Save processed data
        output_file = os.path.join(self.processed_dir, f"processed_{filename}")
        data.to_csv(output_file, index=False)
        logger.info(f"Processed climate data saved to {output_file}")
        
        return data