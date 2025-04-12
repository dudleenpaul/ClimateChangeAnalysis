import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_temperature_patterns(csv_path: str, n_clusters: int = 3):
    """
    Perform clustering anaylsis on climate data using K-means algorithm and visualize results.

    Args:
        csv_path (str): string representation of filepath containing the CSV temperature data
    Kwargs:
        n_clusters (int): number of clusters/groups
    Returns:
        Modified DataFrame with the additonal 'Cluster' column to show which cluster it belongs to
    """
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df[['Year', 'Mean']].values  # NOAA format
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)

    # Plot
    plt.scatter(df['Year'], df['Mean'], c=df['Cluster'], cmap='viridis')
    plt.title("Temperature Clustering Over Years")
    plt.xlabel("Year")
    plt.ylabel("Mean Temp")
    plt.grid(True)
    plt.show()

    return df
