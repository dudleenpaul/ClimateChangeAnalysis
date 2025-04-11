import pandas as pd
import numpy as np
from src.visualizer import (
    plot_predictions, plot_temparature_trends,
    plot_anomalies, plot_clusters, create_animated_visualization
)
from src.data_processor import DataProcessor
from src.algorithms import custom_clustering, detect_anomalies

#load and clean data
processor = DataProcessor("data/climate_data.csv")
processor.load_data()
processor.clean_data()
X, y = processor.get_features_and_target()

# 1. Predictions plot (just dummy values for now)
y_pred = y + np.random.normal(0, 0.1, size=len(y))
plot_predictions(y, y_pred)

# 2. Temperature trend line
data = processor.data.copy()
plot_temparature_trends(data)

# 3. Clustering
labels = custom_clustering(X, n_clusters=3)
plot_clusters(X, labels, feature_names=["Year", "Month"])

# 4. Anomalies
anomalies = detect_anomalies(y)
plot_anomalies(y, anomalies)

# 5. Animated Visualization
create_animated_visualization(data)

