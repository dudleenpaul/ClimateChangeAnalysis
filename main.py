#!/usr/bin/env python3
"""
Climate Change Impact Analyzer - Main script with Berkeley Earth data support

This script serves as the entry point for the Climate Change Impact Analyzer tool.
It provides a command-line interface to analyze climate data from Berkeley Earth.
"""

import sys
import os
import logging
import argparse
from src.simple_climate_data_processor import SimpleClimateDataProcessor
from src.ml_models import ClimateMLModels
from src.visualization import ClimateVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CLI with Berkeley Earth data support."""
    parser = argparse.ArgumentParser(description='Climate Change Impact Analyzer')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Data processing command
    data_parser = subparsers.add_parser('data', help='Data processing operations')
    data_parser.add_argument('--file', default='climate_data.csv', help='Data file to process')
    data_parser.add_argument('--operation', choices=['load', 'process'], 
                            default='process', help='Data operation to perform')
    
    # ML models command
    ml_parser = subparsers.add_parser('ml', help='Machine learning operations')
    ml_parser.add_argument('--file', default='processed_climate_data.csv', help='Data file to use')
    ml_parser.add_argument('--operation', choices=['predict', 'anomaly'], 
                          required=True, help='ML operation to perform')
    ml_parser.add_argument('--years-ahead', type=int, default=10, help='Years to predict ahead')
    
    # Visualization command
    viz_parser = subparsers.add_parser('viz', help='Visualization operations')
    viz_parser.add_argument('--file', default='processed_climate_data.csv', help='Data file to visualize')
    viz_parser.add_argument('--type', choices=['time', 'seasonal', 'decade', 'prediction', 'anomaly'], 
                           required=True, help='Type of visualization')
    viz_parser.add_argument('--time-scale', choices=['monthly', 'yearly', 'seasonal'],
                           default='yearly', help='Time scale for time series')
    viz_parser.add_argument('--decade', help='Decade for seasonal analysis')
    viz_parser.add_argument('--save-path', help='Path to save visualization')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='127.0.0.1', help='Host to run the web server on')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    
    args = parser.parse_args()
    
    if args.command == 'data':
        process_data(args)
    elif args.command == 'ml':
        run_ml_analysis(args)
    elif args.command == 'viz':
        create_visualization(args)
    elif args.command == 'web':
        run_web_interface(args)
    else:
        parser.print_help()

def process_data(args):
    """Process climate data."""
    logger.info(f"Processing data file: {args.file}")
    
    data_processor = SimpleClimateDataProcessor()
    
    if args.operation == 'load':
        data = data_processor.load_climate_data(args.file)
        logger.info(f"Data loaded. Shape: {data.shape}")
        print(data.head())
        
    elif args.operation == 'process':
        data = data_processor.preprocess_pipeline(args.file)
        logger.info(f"Data preprocessing pipeline completed. Shape: {data.shape}")
        print(data.head())

def run_ml_analysis(args):
    """Run machine learning analysis on climate data."""
    logger.info(f"Running ML analysis: {args.operation}")
    
    # Load data
    data_processor = SimpleClimateDataProcessor()
    data_path = os.path.join('data', 'processed', args.file)
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        # If processed file doesn't exist, try to process raw data
        logger.info(f"Processed file not found: {data_path}. Processing raw data.")
        data = data_processor.preprocess_pipeline(args.file.replace('processed_', ''))
    
    ml_models = ClimateMLModels()
    
    if args.operation == 'predict':
        # Prepare yearly data for prediction
        yearly_data = data.groupby('Year')['Temperature'].mean().reset_index()
        
        # Train prediction model
        metrics = ml_models.train_prediction_model(
            yearly_data,
            target='Temperature',
            features=['Year']
        )
        
        logger.info(f"Prediction model trained. R²: {metrics.get('r2', 'N/A')}")
        
        # Create future data
        last_year = yearly_data['Year'].max()
        future_years = np.arange(last_year + 1, last_year + args.years_ahead + 1)
        
        future_data = pd.DataFrame()
        future_data['Year'] = future_years
        
        # Predict future values
        predictions = ml_models.predict_future(future_data, args.years_ahead)
        
        print("Future Predictions:")
        print(predictions)
        
    elif args.operation == 'anomaly':
        # Prepare yearly data for anomaly detection
        yearly_data = data.groupby('Year')['Temperature'].mean().reset_index()
        
        # Detect anomalies
        anomaly_data = ml_models.detect_anomalies(
            yearly_data,
            features=['Temperature']
        )
        
        anomaly_count = anomaly_data['is_anomaly'].sum()
        logger.info(f"Anomaly detection complete. Found {anomaly_count} anomalies.")
        
        print("Detected Anomalies:")
        print(anomaly_data[anomaly_data['is_anomaly']])

def create_visualization(args):
    """Create visualization for climate data."""
    logger.info(f"Creating visualization: {args.type}")
    
    # Load data
    data_processor = SimpleClimateDataProcessor()
    data_path = os.path.join('data', 'processed', args.file)
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        # If processed file doesn't exist, try to process raw data
        logger.info(f"Processed file not found: {data_path}. Processing raw data.")
        data = data_processor.preprocess_pipeline(args.file.replace('processed_', ''))
    
    visualizer = ClimateVisualizer()
    
    if args.type == 'time':
        # Prepare data based on time scale
        if args.time_scale == 'yearly':
            yearly_data = data.groupby('Year')['Temperature'].mean().reset_index()
            plot_data = yearly_data
            x_col = 'Year'
            group_col = None
        elif args.time_scale == 'seasonal':
            seasonal_data = data.groupby(['Year', 'season'])['Temperature'].mean().reset_index()
            plot_data = seasonal_data
            x_col = 'Year'
            group_col = 'season'
        else:  # monthly
            plot_data = data
            x_col = 'date'
            group_col = None
            
        visualizer.plot_time_series(
            plot_data,
            x_col=x_col,
            y_col='Temperature',
            title=f'Temperature Over Time ({args.time_scale.capitalize()})',
            group_col=group_col,
            save_path=args.save_path or f"temperature_{args.time_scale}.png"
        )
        
    elif args.type == 'seasonal':
        # Filter data by decade if specified
        filtered_data = data
        if args.decade and args.decade.isdigit():
            decade_int = int(args.decade)
            filtered_data = data[data['decade'] == decade_int]
        
        # Aggregate by season
        seasonal_data = filtered_data.groupby('season')['Temperature'].mean().reset_index()
        
        # Set a specific order for seasons
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_data['season'] = pd.Categorical(seasonal_data['season'], categories=season_order, ordered=True)
        seasonal_data = seasonal_data.sort_values('season')
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='season', y='Temperature', data=seasonal_data, ax=ax)
        
        title = f'Seasonal Temperature Comparison'
        if args.decade:
            title += f' ({args.decade}s)'
        ax.set_title(title)
        
        plt.savefig(args.save_path or 'seasonal_comparison.png')
        logger.info(f"Seasonal comparison visualization created")
        
    elif args.type == 'decade':
        # Aggregate by decade
        decade_data = data.groupby('decade')['Temperature'].mean().reset_index()
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(decade_data['decade'], decade_data['Temperature'])
        
        # Add trend line
        x = decade_data['decade']
        y = decade_data['Temperature']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8)
        
        ax.set_title('Temperature by Decade')
        ax.set_xlabel('Decade')
        ax.set_ylabel('Temperature')
        
        plt.savefig(args.save_path or 'decade_comparison.png')
        logger.info(f"Decade comparison visualization created")
        
    elif args.type == 'prediction':
        # Prepare yearly data for prediction
        yearly_data = data.groupby('Year')['Temperature'].mean().reset_index()
        
        # Train model
        ml_models = ClimateMLModels()
        ml_models.train_prediction_model(
            yearly_data,
            target='Temperature',
            features=['Year']
        )
        
        # Create future data
        last_year = yearly_data['Year'].max()
        years_ahead = args.years_ahead or 10
        future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
        
        future_data = pd.DataFrame()
        future_data['Year'] = future_years
        
        # Predict future values
        predictions = ml_models.predict_future(future_data, years_ahead)
        
        # Create visualization
        visualizer.plot_prediction_comparison(
            yearly_data,
            predictions,
            x_col='Year',
            actual_col='Temperature',
            predicted_col='predicted_Temperature',
            title='Historical and Predicted Temperature',
            save_path=args.save_path or 'temperature_prediction.png'
        )
        
    elif args.type == 'anomaly':
        # Prepare yearly data for anomaly detection
        yearly_data = data.groupby('Year')['Temperature'].mean().reset_index()
        
        # Detect anomalies
        ml_models = ClimateMLModels()
        anomaly_data = ml_models.detect_anomalies(
            yearly_data,
            features=['Temperature']
        )
        
        # Create visualization
        visualizer.plot_anomalies(
            anomaly_data,
            x_col='Year',
            y_col='Temperature',
            title='Temperature Anomaly Detection',
            save_path=args.save_path or 'temperature_anomalies.png'
        )

def run_web_interface(args):
    """Run the web interface for the climate analyzer."""
    logger.info(f"Starting web interface on {args.host}:{args.port}")
    
    # Import the web app
    from src.web_app import app
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    # Make imports available
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    main()