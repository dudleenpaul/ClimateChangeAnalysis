"""
Utilities for testing the Climate Change Impact Analyzer.
"""

import numpy as np
import pandas as pd
import os

def generate_test_data(output_path, start_year=1950, end_year=2022):
    """
    Generate test climate data for testing.
    
    Args:
        output_path: Path to save the test data
        start_year: First year of data
        end_year: Last year of data
        
    Returns:
        DataFrame containing the test data
    """
    # Create random data with trend and seasonality
    np.random.seed(42)
    years = np.repeat(np.arange(start_year, end_year + 1), 12)
    months = np.tile(np.arange(1, 13), end_year - start_year + 1)
    
    # Create temperature data with trend and seasonal patterns
    base_temp = 0.0
    trend = 0.01 * (years - start_year)  # Warming trend
    seasonal = 0.5 * np.sin(2 * np.pi * (months - 6) / 12)  # Seasonal cycle
    noise = np.random.normal(0, 0.2, len(years))  # Random variations
    
    temps = base_temp + trend + seasonal + noise
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'Year': years,
        'Month': months,
        'Temperature': temps
    })
    
    # Save to CSV if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        test_data.to_csv(output_path, index=False)
    
    return test_data

def compare_dataframes(df1, df2, columns_to_compare=None, tolerance=1e-6):
    """
    Compare two DataFrames for approximate equality.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        columns_to_compare: List of columns to compare (None for all common columns)
        tolerance: Tolerance for numeric comparisons
        
    Returns:
        Tuple of (is_equal, differences dictionary)
    """
    # Check shapes
    if df1.shape != df2.shape:
        return False, {'shape': (df1.shape, df2.shape)}
    
    # Determine columns to compare
    if columns_to_compare is None:
        columns_to_compare = list(set(df1.columns) & set(df2.columns))
    
    # Check columns
    if not set(columns_to_compare).issubset(set(df1.columns)):
        return False, {'missing_columns_df1': set(columns_to_compare) - set(df1.columns)}
    
    if not set(columns_to_compare).issubset(set(df2.columns)):
        return False, {'missing_columns_df2': set(columns_to_compare) - set(df2.columns)}
    
    # Compare data
    differences = {}
    for col in columns_to_compare:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            # For numeric columns, compare with tolerance
            if not np.allclose(df1[col], df2[col], atol=tolerance, equal_nan=True):
                diff_indices = ~np.isclose(df1[col], df2[col], atol=tolerance, equal_nan=True)
                differences[col] = {
                    'type': 'numeric',
                    'indices': np.where(diff_indices)[0],
                    'values1': df1.loc[diff_indices, col].values,
                    'values2': df2.loc[diff_indices, col].values
                }
        else:
            # For non-numeric columns, compare exactly
            if not df1[col].equals(df2[col]):
                diff_indices = df1[col] != df2[col]
                differences[col] = {
                    'type': 'non-numeric',
                    'indices': np.where(diff_indices)[0],
                    'values1': df1.loc[diff_indices, col].values,
                    'values2': df2.loc[diff_indices, col].values
                }
    
    return len(differences) == 0, differences