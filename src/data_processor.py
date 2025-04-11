import pandas as pd
import numpy as np
from typing import Tuple

class DataProcessor:
    def __init__(self, file_path: str):
        """
        Initalize the DataProcessor class object.

        Args:
            file_path (str): a string representation of the filepath containing the CSV data
        """
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load climate data from CSV file.
        
        Returns:
            a pandas DataFrame object with the loaded data (2d data struct with labeled axes)
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def clean_data(self) -> pd.DataFrame:
        """
        Remove rows with missing values and normalize all numeric columns.
        
        Returns:
            a pandas DataFrame object with the cleaned data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.data.dropna(inplace=True)

        numeric_cols = ['year', 'month', 'temperature']
        for col in numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            if std == 0:
                std = 1
            self.data[col] = (self.data[col] - mean) / std

        return self.data

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into features (year, month) and target (temperature).
        
        Returns:
            a tuple containing: a ndarray with the year and month values, a ndarray with the temperature values
        """
        if self.data is None:
            raise ValueError("Data not loaded or cleaned.")

        X = self.data[['year', 'month']].values
        y = self.data['temperature'].values

        return X, y
