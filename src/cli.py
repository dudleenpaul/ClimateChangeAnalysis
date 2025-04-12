import argparse
import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.ml_models import ClimateMLModels
from src.visualization import ClimateVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CliApp:
    """
    Command-line interface for the Climate Change Impact Analyzer.
    """
    
    def __init__(self):
        """Initialize CLI application with required components."""
        self.data_processor = DataProcessor()
        self.ml_models = ClimateMLModels()
        self.visualizer = ClimateVisualizer()
        self.data = None
    
    def parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Climate Change Impact Analyzer')
        
        # Add subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Data processing command
        data_parser = subparsers.add_parser('data', help='Data processing operations')
        data_parser.add_argument('--file', required=True, help='Data file to process')
        data_parser.add_argument('--operation', choices=['load', 'clean', 'normalize', 'pipeline'], 
                                default='pipeline', help='Data operation to perform')
        
        # ML models command
        ml_parser = subparsers.add_parser('ml', help='Machine learning operations')
        ml_parser.add_argument('--file', required=True, help='Data file to use')
        ml_parser.add_argument('--operation', choices=['predict', 'cluster', 'anomaly'], 
                              required=True, help='ML operation to perform')
        ml_parser.add_argument('--target', help='Target variable for prediction')
        ml_parser.add_argument('--features', nargs='+', help='Features to use for analysis')
        ml_parser.add_argument('--years-ahead', type=int, default=5, help='Years to predict ahead')
        ml_parser.add_argument('--n-clusters', type=int, default=3, help='Number of clusters')
        
        # Visualization command
        viz_parser = subparsers.add_parser('viz', help='Visualization operations')
        viz_parser.add_argument('--file', required=True, help='Data file to visualize')
        viz_parser.add_argument('--type', choices=['time', 'heatmap', 'regional', 'prediction', 'anomaly'], 
                               required=True, help='Type of visualization')
        viz_parser.add_argument('--x-col', help='X-axis column')
        viz_parser.add_argument('--y-col', help='Y-axis column')
        viz_parser.add_argument('--group-col', help='Column to group by')
        viz_parser.add_argument('--title', help='Plot title')
        viz_parser.add_argument('--save-path', help='Path to save visualization')
        
        # Run full pipeline
        pipeline_parser = subparsers.add_parser('pipeline', help='Run full analysis pipeline')
        pipeline_parser.add_argument('--file', required=True, help='Data file to process')
        pipeline_parser.add_argument('--target', required=True, help='Target variable for prediction')
        pipeline_parser.add_argument('--features', nargs='+', help='Features to use for analysis')
        pipeline_parser.add_argument('--years-ahead', type=int, default=5, help='Years to predict ahead')
        
        return parser.parse_args()
    
    def run(self, args):
        """
        Run the Climate Change Impact Analyzer with the provided arguments.
        
        Args:
            args: Command line arguments
        """
        logger.info(f"Running command: {args.command}")
        
        # Execute command
        if args.command == 'data':
            self._run_data_command(args)
        elif args.command == 'ml':
            self._run_ml_command(args)
        elif args.command == 'viz':
            self._run_viz_command(args)
        elif args.command == 'pipeline':
            self._run_pipeline_command(args)
        else:
            logger.error("Invalid command. Use --help for usage information.")
    
    def _run_data_command(self, args):
        """Execute data processing command."""
        if args.operation == 'load':
            self.data = self.data_processor.load_data(args.file)
            logger.info(f"Data loaded. Shape: {self.data.shape}")
            print(self.data.head())
            
        elif args.operation == 'clean':
            self.data = self.data_processor.load_data(args.file)
            self.data = self.data_processor.clean_data()
            logger.info(f"Data cleaned. Shape: {self.data.shape}")
            print(self.data.head())
            
        elif args.operation == 'normalize':
            self.data = self.data_processor.load_data(args.file)
            self.data = self.data_processor.clean_data()
            self.data = self.data_processor.normalize_data()
            logger.info(f"Data normalized. Shape: {self.data.shape}")
            print(self.data.head())
            
        elif args.operation == 'pipeline':
            self.data = self.data_processor.preprocess_pipeline(args.file)
            logger.info(f"Data preprocessing pipeline completed. Shape: {self.data.shape}")
            print(self.data.head())
    
    def _run_ml_command(self, args):
        """Execute machine learning command."""
        # Load data first
        self.data = self.data_processor.load_data(args.file)
        
        if args.operation == 'predict':
            if not args.target:
                logger.error("Target variable must be specified for prediction.")
                return
                
            # Train prediction model
            metrics = self.ml_models.train_prediction_model(
                self.data,
                target=args.target,
                features=args.features
            )
            
            # Print performance metrics
            print("Prediction Model Performance:")
            for key, value in metrics.items():
                if key == 'feature_importance':
                    print(f"\nFeature Importance:")
                    for feat, imp in sorted(value.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {feat}: {imp:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Make future predictions
            last_year = self.data['year'].max() if 'year' in self.data.columns else 0
            future_years = np.arange(last_year + 1, last_year + args.years_ahead + 1)
            
            # Create future data (simple copy of last data point)
            future_data = pd.DataFrame()
            for feature in self.ml_models.prediction_features:
                if feature == 'year':
                    future_data['year'] = future_years
                else:
                    # Use the last value for other features
                    future_data[feature] = self.data[feature].iloc[-1]
            
            # Predict future values
            predictions = self.ml_models.predict_future(future_data, args.years_ahead)
            print("\nFuture Predictions:")
            print(predictions)
            
        elif args.operation == 'cluster':
            if not args.features:
                logger.error("Features must be specified for clustering.")
                return
                
            # Cluster regions
            clustered_data = self.ml_models.cluster_regions(
                self.data,
                features=args.features,
                n_clusters=args.n_clusters
            )
            
            # Print cluster info
            cluster_counts = clustered_data['cluster'].value_counts()
            print("Cluster Distribution:")
            for cluster, count in cluster_counts.items():
                print(f"  Cluster {cluster}: {count} points")
            
            # Display cluster characteristics
            print("\nCluster Characteristics:")
            cluster_means = clustered_data.groupby('cluster')[args.features].mean()
            print(cluster_means)
            
        elif args.operation == 'anomaly':
            if not args.features:
                logger.error("Features must be specified for anomaly detection.")
                return
                
            # Detect anomalies
            anomaly_data = self.ml_models.detect_anomalies(
                self.data,
                features=args.features
            )
            
            # Print anomaly info
            anomaly_count = anomaly_data['is_anomaly'].sum()
            print(f"Detected {anomaly_count} anomalies out of {len(anomaly_data)} points.")
            
            # Display anomalies
            print("\nAnomaly Data Sample:")
            print(anomaly_data[anomaly_data['is_anomaly']].head())
    
    def _run_viz_command(self, args):
        """Execute visualization command."""
        # Load data first
        self.data = self.data_processor.load_data(args.file)
        
        # Set default values if not provided
        title = args.title or f"Climate Data Visualization"
        save_path = args.save_path or f"{args.type}_plot.png"
        
        if args.type == 'time':
            if not args.x_col or not args.y_col:
                logger.error("X and Y columns must be specified for time series visualization.")
                return
                
            self.visualizer.plot_time_series(
                self.data,
                x_col=args.x_col,
                y_col=args.y_col,
                title=title,
                group_col=args.group_col,
                save_path=save_path
            )
            logger.info(f"Time series visualization created and saved to {save_path}")
            
        elif args.type == 'heatmap':
            if not args.x_col or not args.y_col:
                logger.error("X and Y columns must be specified for heatmap visualization.")
                return
                
            self.visualizer.plot_heatmap(
                self.data,
                value_col=args.y_col,
                x_col=args.x_col,
                y_col=args.group_col or 'region',
                title=title,
                save_path=save_path
            )
            logger.info(f"Heatmap visualization created and saved to {save_path}")
            
        elif args.type == 'regional':
            if not args.y_col or not args.group_col:
                logger.error("Y column and group column must be specified for regional visualization.")
                return
                
            regions = self.data[args.group_col].unique()
            self.visualizer.plot_regional_comparison(
                self.data,
                value_col=args.y_col,
                regions=regions,
                region_col=args.group_col,
                title=title,
                save_path=save_path
            )
            logger.info(f"Regional comparison visualization created and saved to {save_path}")
            
        elif args.type == 'prediction':
            if not args.x_col or not args.y_col:
                logger.error("X and Y columns must be specified for prediction visualization.")
                return
                
            # First 80% for historical data
            split_idx = int(len(self.data) * 0.8)
            historical_data = self.data.iloc[:split_idx].copy()
            
            # Train a model on historical data
            features = args.features or [col for col in self.data.columns if col != args.y_col]
            self.ml_models.train_prediction_model(
                historical_data,
                target=args.y_col,
                features=features
            )
            
            # Predict on test data
            future_data = self.data.iloc[split_idx:].copy()
            predictions = self.ml_models.predict_future(future_data)
            
            self.visualizer.plot_prediction_comparison(
                historical_data,
                predictions,
                x_col=args.x_col,
                actual_col=args.y_col,
                title=title,
                save_path=save_path
            )
            logger.info(f"Prediction comparison visualization created and saved to {save_path}")
            
        elif args.type == 'anomaly':
            if not args.x_col or not args.y_col:
                logger.error("X and Y columns must be specified for anomaly visualization.")
                return
                
            # Detect anomalies
            features = args.features or [args.x_col, args.y_col]
            anomaly_data = self.ml_models.detect_anomalies(
                self.data,
                features=features
            )
            
            self.visualizer.plot_anomalies(
                anomaly_data,
                x_col=args.x_col,
                y_col=args.y_col,
                title=title,
                save_path=save_path
            )
            logger.info(f"Anomaly visualization created and saved to {save_path}")
    
    def _run_pipeline_command(self, args):
        """Execute full analysis pipeline."""
        logger.info("Running full analysis pipeline")
        
        # 1. Data preprocessing
        self.data = self.data_processor.preprocess_pipeline(args.file)
        
        # 2. Train prediction model
        features = args.features or [col for col in self.data.columns 
                                   if col != args.target and col != 'year']
        
        metrics = self.ml_models.train_prediction_model(
            self.data,
            target=args.target,
            features=features
        )
        
        print("Prediction Model Performance:")
        for key, value in metrics.items():
            if key == 'feature_importance':
                print(f"\nFeature Importance:")
                for feat, imp in sorted(value.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {feat}: {imp:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 3. Make future predictions
        last_year = self.data['year'].max() if 'year' in self.data.columns else 0
        future_years = np.arange(last_year + 1, last_year + args.years_ahead + 1)
        
        # Create future data (simple copy of last data point)
        future_data = pd.DataFrame()
        for feature in self.ml_models.prediction_features:
            if feature == 'year':
                future_data['year'] = future_years
            else:
                # Use the last value for other features
                future_data[feature] = self.data[feature].iloc[-1]
        
        # Predict future values
        predictions = self.ml_models.predict_future(future_data, args.years_ahead)
        
        # 4. Create visualizations
        if 'year' in self.data.columns:
            # Time series plot
            self.visualizer.plot_time_series(
                self.data,
                x_col='year',
                y_col=args.target,
                title=f"{args.target} Over Time",
                save_path=f"time_series_{args.target}.png"
            )
            
            # Prediction plot
            self.visualizer.plot_prediction_comparison(
                self.data,
                predictions,
                x_col='year',
                actual_col=args.target,
                title=f"Historical and Predicted {args.target}",
                save_path=f"prediction_{args.target}.png"
            )
        
        # Anomaly detection
        anomaly_data = self.ml_models.detect_anomalies(
            self.data,
            features=features
        )
        
        # Pick two features for anomaly visualization
        x_feature = features[0] if features else self.data.columns[0]
        y_feature = args.target
        
        self.visualizer.plot_anomalies(
            anomaly_data,
            x_col=x_feature,
            y_col=y_feature,
            title=f"Anomaly Detection",
            save_path="anomaly_detection.png"
        )
        
        logger.info("Full analysis pipeline completed")

def main():
    """Main entry point for the CLI."""
    app = CliApp()
    args = app.parse_args()
    app.run(args)

if __name__ == "__main__":
    main()