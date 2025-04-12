import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Union
import os
import logging
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClimateVisualizer:
    """
    Module for visualizing climate data and analysis results.
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default styling
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Custom color palette for climate data
        self.temp_cmap = LinearSegmentedColormap.from_list(
            'temp_cmap', ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027', '#a50026']
        )
    
    def plot_time_series(self, 
                        data: pd.DataFrame, 
                        x_col: str, 
                        y_col: str, 
                        title: str, 
                        group_col: Optional[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series data with optional grouping.
        
        Args:
            data: DataFrame containing time series data
            x_col: Column to use for x-axis (typically 'year')
            y_col: Column to use for y-axis
            title: Plot title
            group_col: Optional column to group by (e.g., 'region')
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        logger.info(f"Creating time series plot for {y_col} vs {x_col}")
        
        fig, ax = plt.subplots()
        
        if group_col and group_col in data.columns:
            groups = data[group_col].unique()
            for group in groups:
                group_data = data[data[group_col] == group]
                ax.plot(group_data[x_col], group_data[y_col], label=group)
            ax.legend(title=group_col)
        else:
            ax.plot(data[x_col], data[y_col])
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        
        # Add trend line
        if len(data) > 1:
            try:
                z = np.polyfit(data[x_col], data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, label="Trend")
                if group_col is None:
                    ax.legend()
            except Exception as e:
                logger.warning(f"Could not add trend line: {e}")
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300)
            logger.info(f"Saved time series plot to {full_path}")
        
        return fig
    
    def plot_heatmap(self, 
                    data: pd.DataFrame, 
                    value_col: str, 
                    x_col: str, 
                    y_col: str, 
                    title: str,
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            data: DataFrame containing data to visualize
            value_col: Column containing values to plot
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        logger.info(f"Creating heatmap for {value_col}")
        
        # Pivot data into heatmap format
        pivot_data = data.pivot_table(index=y_col, columns=x_col, values=value_col)
        
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(
            pivot_data, 
            cmap=self.temp_cmap, 
            ax=ax,
            annot=False, 
            linewidths=0.5
        )
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300)
            logger.info(f"Saved heatmap to {full_path}")
        
        return fig
    
    def plot_regional_comparison(self, 
                               data: pd.DataFrame, 
                               value_col: str, 
                               regions: List[str],
                               region_col: str = 'region',
                               title: str = 'Regional Comparison',
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart comparing values across regions.
        
        Args:
            data: DataFrame containing regional data
            value_col: Column containing values to compare
            regions: List of regions to include
            region_col: Column containing region identifiers
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        logger.info(f"Creating regional comparison for {value_col}")
        
        # Filter data for specified regions
        filtered_data = data[data[region_col].isin(regions)].copy()
        
        if filtered_data.empty:
            logger.warning(f"No data found for specified regions: {regions}")
            return None
        
        # Group by region and calculate mean
        regional_means = filtered_data.groupby(region_col)[value_col].mean().reset_index()
        
        fig, ax = plt.subplots()
        sns.barplot(x=region_col, y=value_col, data=regional_means, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel(region_col)
        ax.set_ylabel(f'Average {value_col}')
        
        # Rotate x labels if there are many regions
        if len(regions) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300)
            logger.info(f"Saved regional comparison to {full_path}")
        
        return fig
    
    def plot_prediction_comparison(self, 
                                 historical_data: pd.DataFrame,
                                 predicted_data: pd.DataFrame,
                                 x_col: str = 'year',
                                 actual_col: str = 'temperature',
                                 predicted_col: str = 'predicted_temperature',
                                 title: str = 'Historical vs Predicted Values',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of historical vs predicted values.
        
        Args:
            historical_data: DataFrame with historical data
            predicted_data: DataFrame with predicted values
            x_col: Column to use for x-axis (typically 'year')
            actual_col: Column name for actual values
            predicted_col: Column name for predicted values
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        logger.info("Creating prediction comparison plot")
        
        fig, ax = plt.subplots()
        
        # Plot historical data
        ax.plot(historical_data[x_col], historical_data[actual_col], label='Historical', color='blue')
        
        # Plot predicted data
        ax.plot(predicted_data[x_col], predicted_data[predicted_col], label='Predicted', 
                color='red', linestyle='--')
        
        # Add shaded area for predictions to indicate uncertainty
        ax.fill_between(
            predicted_data[x_col], 
            predicted_data[predicted_col] * 0.9, 
            predicted_data[predicted_col] * 1.1, 
            color='red', 
            alpha=0.2
        )
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(actual_col)
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300)
            logger.info(f"Saved prediction comparison to {full_path}")
        
        return fig
    
    def create_animated_plot(self, 
                           data: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           time_col: str,
                           title: str,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create an animated plot showing change over time.
        
        Args:
            data: DataFrame containing data to animate
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            time_col: Column containing time values for animation frames
            title: Plot title
            save_path: Optional path to save the animation
            
        Returns:
            Figure object
        """
        logger.info(f"Creating animated plot for {y_col} vs {x_col} over {time_col}")
        
        fig, ax = plt.subplots()
        
        # Get unique time values
        time_values = sorted(data[time_col].unique())
        
        # Initialize empty line
        line, = ax.plot([], [], 'b-')
        
        # Set axis limits
        ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(data[y_col].min(), data[y_col].max())
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        
        # Title with placeholder for time value
        title_obj = ax.set_title(f"{title} - {time_values[0]}")
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(i):
            time_val = time_values[i]
            frame_data = data[data[time_col] == time_val]
            line.set_data(frame_data[x_col], frame_data[y_col])
            title_obj.set_text(f"{title} - {time_val}")
            return line, title_obj
        
        ani = FuncAnimation(fig, animate, frames=len(time_values),
                           init_func=init, blit=True, interval=200)
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            ani.save(full_path, writer='pillow', fps=2)
            logger.info(f"Saved animation to {full_path}")
        
        return fig
    
    def plot_anomalies(self, 
                      data: pd.DataFrame, 
                      x_col: str, 
                      y_col: str, 
                      anomaly_col: str = 'is_anomaly',
                      title: str = 'Anomaly Detection',
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data points with anomalies highlighted.
        
        Args:
            data: DataFrame containing data with anomaly flags
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            anomaly_col: Column containing anomaly flags (True/False)
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        logger.info(f"Creating anomaly plot for {y_col} vs {x_col}")
        
        fig, ax = plt.subplots()
        
        # Plot normal points
        normal_data = data[~data[anomaly_col]]
        ax.scatter(normal_data[x_col], normal_data[y_col], 
                  label='Normal', color='blue', alpha=0.6)
        
        # Plot anomalies
        anomaly_data = data[data[anomaly_col]]
        ax.scatter(anomaly_data[x_col], anomaly_data[y_col], 
                  label='Anomaly', color='red', alpha=0.8)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300)
            logger.info(f"Saved anomaly plot to {full_path}")
        
        return fig