import os
from data_processor import DataProcessor
from algorithms import CustomTemperaturePredictor, custom_clustering, detect_anomalies
from visualizer import (
    plot_predictions, 
    plot_temparature_trends, 
    plot_anomalies, 
    plot_clusters,
    create_animated_visualization
)

#from src.data_processor import DataProcessor
#from src.algorithms import CustomTemperaturePredictor
#from src.visualizer import plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_results_directory():
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created a results directory for all outputs.")
    
def run_pipeline(file_path: str):
    print(f"Starting the climate change impact analysis on {file_path}...")
    
    processor = DataProcessor(file_path)
    #processor.load_data()
    raw_data = processor.load_data()
    #processor.clean_data()
   # X, y = processor.get_features_and_target()
    raw_data_copy = raw_data.copy()
    
    normalized_data = processor.clean_data()
    print("The data has been cleaned and normalized.")
    
    X, y = processor.get_features_and_target()
    
    full_X, full_y = X.copy(), y.copy()
    
    print("\n--- Temparature Prediction ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CustomTemperaturePredictor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    plot_predictions(y_test, y_pred)
    
    print("\n--- Climate Data Cluster ---")
    cluster_labels = custom_clustering(X, n_clusters = 3)
    
    unique_labels, counts = np.unique(cluster_labels, return_counts = True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples")
        
    plot_clusters(X, cluster_labels, ["Normalized Year", "Normalized Month"])
    
    print("\n--- Anomaly Detection ---")
    
    yearly_data = raw_data_copy.sort_values(['year', 'month'])
    time_series = yearly_data['temperature'].values
    years = yearly_data['year'].values
    
    anomalies = detect_anomalies(time_series, window_size = 12, threshold = 2.0)
    
    anomaly_indices = plot_anomalies(time_series, anomalies, years)
    
    print("\n--- Temperature Trend Analysis ---")
    yearly_avg = plot_temparature_trends(raw_data_copy)
    
    average_change = yearly_avg['temparature'].diff().mean()
    print(f"The average temperature change per year: {average_change:.6f}°C")
    
    print("\n--- Creating Temperature Animation ---")
    try:
        create_animated_visualization(raw_data_copy)
    except Exception as e:
        print(f"Could not create the animation: {e}")
        
    print("\nThe analysis is complete! The results will be found in the results directory.")
    
    return {
        "prediction_mse": mse,
        "cluster_counts": dict(zip(unique_labels.tolist(), counts.tolist())),
        "anomaly_count": len(anomaly_indices),
        "avg_temp_change": average_change
    }

if __name__ == "__main__":
    ensure_results_directory()
    run_pipeline("data/climate_data.csv")

