""" visualizer.py
Contains methods for visualizing temperature data

Methods:
    plot_predictions(y_true, y_pred): plot actual vs predicted temperature values
    plot_temparature_trends(data): <desc>
    plot_anomalies(time_series, anomalies, years): <desc>
    plot_clusters(data, labels, feature_names): <desc>
    create_animated_visualization(data): <desc>
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
	"""
	Plot actual vs predicted temperature values.

	Args: 
		y_true (ndarray): actual temperature values
		y_pred (ndarray): predicted temperature values
	"""
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true)), y_true, label='Actual', alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature (normalized)')
    plt.title('Predicted vs. Actual Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_temparature_trends(data: pd.DataFrame):
	"""
	Plots the average temperature by year using a line graph

	Additionally, saves the plot to 'results/temperature_trends.png'

	Args:
		data (DataFrame): a pandas DataFrame of the temperature data
	Returns:
		a DataFrame containing the average temperature per year
	"""
    plt.figure(figsize=(12, 6))
    
    yearly_average = data.groupby('year')['temperature'].mean().reset_index()
    
    #creating the plot
    plt.plot(yearly_average['year'], yearly_average['temperature'], marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature')
    plt.title('Temperature Trends Over Time')
    plt.grid(True)
    
    z = np.polyfit(yearly_average['year'], yearly_average['temperature'], 1)
    p = np.poly1d(z)
    plt.plot(yearly_average['year'], p(yearly_average['year']), "r--", alpha=0.8, 
             label=f"Trend: {z[0]:.6f}°C/year")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/temperature_trends.png')
    plt.show()
    
    return yearly_average


def plot_anomalies(time_series: np.ndarray, anomalies: np.ndarray, years: Optional[np.ndarray] = None):
	"""
	Plots the full time series of temperature data and anomalies in this data

	Blue line is the full temperature over time (in years)
	Red dots are anomalous temperature values

	Args:
		time_series (ndarray): a numpy array of the temperature data
		anomalies (ndarray): a numpy array of the anomalies
	Kwargs:
		years (ndarray): an optional array of the years
	Returns:
		the indicies of the anomalies in the time series
	"""
    plt.figure(figsize=(14, 7))
    
    x_values = years if years is not None else range(len(time_series))
    
    plt.plot(x_values, time_series, label = 'Temperature', color = 'blue', alpha = 0.7)
    
    anomaly_indices = np.where(anomalies)[0]
    plt.scatter(x_values[anomaly_indices], time_series[anomaly_indices], color = 'red', s = 50, label = 'Anomalies')
    
    plt.title('Temperature Anomalies Detection')
    plt.xlabel('Time' if years is None else 'Year')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/temperature_anomalies.png')
    plt.show()
    
    print(f"Detected {len(anomaly_indices)} anomalies out of {len(time_series)} data points")
    return anomaly_indices


def plot_clusters(data: np.ndarray, labels: np.ndarray, feature_names: List[str]):
    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(10, 8))

    if data.shape[1] >= 2:
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        label=f'Cluster {i}', alpha=0.7)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    
    plt.title(f'Climate Data Clustering ({n_clusters} clusters)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/climate_clusters.png')
    plt.show()


def create_animated_visualization(data: pd.DataFrame):
    pivot_data = data.pivot_table(index='month', columns='year', values='temperature')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    years = sorted(data['year'].unique())
    months = range(1, 13)
    
    line, = ax.plot(months, pivot_data[years[0]], 'o-', lw=2)
    
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(pivot_data.min().min() - 0.5, pivot_data.max().max() + 0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Temperature')
    title = ax.set_title(f'Temperature by Month - Year: {years[0]}')
    
    def update(frame):
        """Update function for animation frame."""
        year = years[frame]
        line.set_ydata(pivot_data[year])
        title.set_text(f'Temperature by Month - Year: {year}')
        return line, title
    
    ani = FuncAnimation(fig, update, frames=len(years), interval=300, blit=True)
    
    # Save the animation
    try:
        ani.save('results/temperature_animation.gif', writer='pillow', fps=2)
        print("Animation saved as 'results/temperature_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.tight_layout()
    plt.show()
