import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detect_temperature_anomalies(csv_path: str, z_thresh: float = 2.0):
    """
    Detect and plot temperature anomalies directly from the CSV file

    Args:
        csv_path (str): the filepath of the CSV file containing temperature data
    Kwargs:
        z_thresh (float): the threshold for a temperature to be considered an anomaly
    Returns:
        a list of anomalies
    """
    df = pd.read_csv(csv_path)
    mean_temp = df['Mean'].mean()
    std_temp = df['Mean'].std()
    df['Z-Score'] = (df['Mean'] - mean_temp) / std_temp
    df['Anomaly'] = df['Z-Score'].abs() > z_thresh

    # Plot anomalies
    plt.plot(df['Year'], df['Mean'], label='Mean Temp')
    plt.scatter(df[df['Anomaly']]['Year'], df[df['Anomaly']]['Mean'], color='red', label='Anomaly')
    plt.title("Anomaly Detection in Global Temperature")
    plt.xlabel("Year")
    plt.ylabel("Mean Temp")
    plt.legend()
    plt.show()

    return df[df['Anomaly']]
