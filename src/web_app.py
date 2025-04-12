from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import sys


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.simple_climate_data_processor import SimpleClimateDataProcessor
from src.ml_models import ClimateMLModels
from src.visualization import ClimateVisualizer

app = Flask(__name__)

# Initialize components
data_processor = SimpleClimateDataProcessor()
ml_models = ClimateMLModels()
visualizer = ClimateVisualizer()

# Global variables
DATA_FILE = 'climate_data.csv'
data = None

def load_data():
    """Load and prepare the climate data."""
    global data
    
    # Check if data file exists
    data_path = os.path.join('..','data', 'raw', DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Climate data file not found: {data_path}")
    
    # Load and process the data
    data = data_processor.preprocess_pipeline(DATA_FILE)
    
    # Add debug information about data size
    data_size_mb = sys.getsizeof(data) / (1024 * 1024)
    print(f"Data loaded: {len(data)} rows, approximately {data_size_mb:.2f} MB in memory")

def get_data():
    """Get or load the climate data as needed (lazy loading)."""
    global data
    try:
        if data is None:
            print("Loading data...")
            try:
                load_data()
                print("Data loaded successfully.")
            except Exception as e:
                print(f"Error loading data: {e}")
                # Create a minimal dataset to prevent crashes
                import pandas as pd
                import numpy as np
                data = pd.DataFrame({
                    'Year': range(1900, 2025),
                    'Temperature': np.random.normal(0, 1, 125),
                    'date': pd.date_range(start='1/1/1900', periods=125, freq='YE')
                })
                data['decade'] = (data['Year'] // 10) * 10
                seasons = ['Winter', 'Spring', 'Summer', 'Fall']
                data['season'] = [seasons[i % 4] for i in range(len(data))]
                print("Using fallback data")
        return data
    except Exception as e:
        print(f"Critical error in get_data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({
            'Year': range(1900, 2025),
            'Temperature': np.random.normal(0, 1, 125),
            'date': pd.date_range(start='1/1/1900', periods=125, freq='Y')
        })
    
def get_year_range():
    """Get the min and max years from the dataset."""
    data_obj = get_data()
    if data_obj is not None:
        min_year = int(data_obj['Year'].min())
        max_year = int(data_obj['Year'].max())
        return min_year, max_year
    return 1900, 2025  # Default range if data not loaded

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/year-range')
def year_range_info():
    """API endpoint to get min and max years in the dataset."""
    try:
        min_year, max_year = get_year_range()
        return jsonify({
            'min_year': min_year,
            'max_year': max_year
        })
    except Exception as e:
        print(f"Error in year_range_info: {e}")
        return jsonify({
            'min_year': 1900,
            'max_year': 2025,
            'error': str(e)
        }), 200  # Return 200 with error info instead of 500
    
@app.route('/api/time-series')
def time_series():
    """API endpoint for time series data."""
    parameter = request.args.get('parameter', 'Temperature')
    time_scale = request.args.get('time_scale', 'monthly')
    start_year = request.args.get('start_year', None)
    end_year = request.args.get('end_year', None)
    
    # Get data using lazy loading
    current_data = get_data()
    
    # Apply year filter if specified
    filtered_data = current_data
    if start_year and start_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] >= int(start_year)]
    if end_year and end_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] <= int(end_year)]
    
    # Prepare data based on time scale
    if time_scale == 'yearly':
        yearly_data = filtered_data.groupby('Year')[parameter].mean().reset_index()
        plot_data = yearly_data
        x_col = 'Year'
        group_col = None
    elif time_scale == 'seasonal':
        seasonal_data = filtered_data.groupby(['Year', 'season'])[parameter].mean().reset_index()
        plot_data = seasonal_data
        x_col = 'Year'
        group_col = 'season'
    else:  # monthly (default)
        plot_data = filtered_data
        x_col = 'date'
        group_col = None
    
    # Create visualization
    fig = visualizer.plot_time_series(
        plot_data,
        x_col=x_col,
        y_col=parameter,
        title=f'{parameter} Over Time ({time_scale.capitalize()})',
        group_col=group_col if time_scale == 'seasonal' else None,
        save_path=None
    )
    
    # Convert plot to base64
    plot_image = plot_to_base64(fig)
    
    # Prepare data for JSON response
    if time_scale == 'yearly':
        response_data = yearly_data.to_dict(orient='records')
    elif time_scale == 'seasonal':
        response_data = seasonal_data.to_dict(orient='records')
    else:
        # For monthly data, convert date to string for JSON serialization
        monthly_data = plot_data[['date', parameter]].copy()
        monthly_data['date'] = monthly_data['date'].dt.strftime('%Y-%m-%d')
        response_data = monthly_data.to_dict(orient='records')
# In /api/time-series
    if current_data is None or current_data.empty:
        print("No valid data available.")
        return jsonify({
            'plot': '',
            'data': [],
            'error': 'No valid data available.'
        }), 500
        
    return jsonify({
        'plot': plot_image,
        'data': response_data
    })

@app.route('/api/seasonal-comparison')
def seasonal_comparison():
    """API endpoint for seasonal comparison data."""
    parameter = request.args.get('parameter', 'Temperature')
    decade = request.args.get('decade', None)
    start_year = request.args.get('start_year', None)
    end_year = request.args.get('end_year', None)
    
    # Get data using lazy loading
    current_data = get_data()
    
    # Apply year filter if specified
    filtered_data = current_data
    if start_year and start_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] >= int(start_year)]
    if end_year and end_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] <= int(end_year)]
    
    # Filter data by decade if specified
    if decade and decade.isdigit():
        decade_int = int(decade)
        filtered_data = filtered_data[filtered_data['decade'] == decade_int]
    
    # Aggregate by season
    seasonal_data = filtered_data.groupby('season')[parameter].mean().reset_index()
    
    # Set a specific order for seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_data['season'] = pd.Categorical(seasonal_data['season'], categories=season_order, ordered=True)
    seasonal_data = seasonal_data.sort_values('season')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='season', y=parameter, data=seasonal_data, ax=ax)
    
    title = f'Seasonal {parameter} Comparison'
    if decade:
        title += f' ({decade}s)'
    if start_year and end_year:
        title += f' ({start_year}-{end_year})'
    ax.set_title(title)
    
    # Convert plot to base64
    plot_image = plot_to_base64(fig)
    
    return jsonify({
        'plot': plot_image,
        'data': seasonal_data.to_dict(orient='records')
    })

@app.route('/api/decade-comparison')
def decade_comparison():
    """API endpoint for comparison across decades."""
    parameter = request.args.get('parameter', 'Temperature')
    start_year = request.args.get('start_year', None)
    end_year = request.args.get('end_year', None)
    
    # Get data using lazy loading
    current_data = get_data()
    
    # Apply year filter if specified
    filtered_data = current_data
    if start_year and start_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] >= int(start_year)]
    if end_year and end_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] <= int(end_year)]
    
    # Aggregate by decade
    decade_data = filtered_data.groupby('decade')[parameter].mean().reset_index()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(decade_data['decade'], decade_data[parameter])
    
    # Add trend line
    x = decade_data['decade']
    y = decade_data[parameter]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    title = f'{parameter} by Decade'
    if start_year and end_year:
        title += f' ({start_year}-{end_year})'
    ax.set_title(title)
    ax.set_xlabel('Decade')
    ax.set_ylabel(parameter)
    
    # Convert plot to base64
    plot_image = plot_to_base64(fig)
    
    return jsonify({
        'plot': plot_image,
        'data': decade_data.to_dict(orient='records')
    })

@app.route('/api/predictions')
def predictions():
    """API endpoint for future predictions."""
    parameter = request.args.get('parameter', 'Temperature')
    years_ahead = int(request.args.get('years_ahead', 10))
    start_year = request.args.get('start_year', None)
    end_year = request.args.get('end_year', None)
    
    # Get data using lazy loading
    current_data = get_data()
    
    # Apply year filter if specified
    filtered_data = current_data
    if start_year and start_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] >= int(start_year)]
    if end_year and end_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] <= int(end_year)]
    
    # Prepare yearly data for prediction
    yearly_data = filtered_data.groupby('Year')[parameter].mean().reset_index()
    
    # Use years as the single feature for simple prediction
    features = ['Year']
    
    # Train model
    ml_models.train_prediction_model(
        yearly_data,
        target=parameter,
        features=features
    )
    
    # Create future data
    last_year = yearly_data['Year'].max()
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
    
    future_data = pd.DataFrame()
    future_data['Year'] = future_years
    
    # Predict future values
    predictions = ml_models.predict_future(future_data, years_ahead)
    if 'Year' not in predictions.columns and 'Year' in future_data.columns:
        predictions['Year'] = future_data['Year']
    # Create visualization
    fig = visualizer.plot_prediction_comparison(
        yearly_data,
        predictions,
        x_col='Year',
        actual_col=parameter,
        predicted_col=f'predicted_{parameter}',
        title=f'Historical and Predicted {parameter}',
        save_path=None
    )
    
    # Convert plot to base64
    plot_image = plot_to_base64(fig)
    
    # Combine historical and prediction data
    historical_list = yearly_data.to_dict(orient='records')
    prediction_list = predictions.to_dict(orient='records')
    
    return jsonify({
        'plot': plot_image,
        'historical_data': historical_list,
        'prediction_data': prediction_list
    })

@app.route('/api/anomalies')
def anomalies():
    """API endpoint for anomaly detection."""
    parameter = request.args.get('parameter', 'Temperature')
    start_year = request.args.get('start_year', None)
    end_year = request.args.get('end_year', None)
    
    # Get data using lazy loading
    current_data = get_data()
    
    # Apply year filter if specified
    filtered_data = current_data
    if start_year and start_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] >= int(start_year)]
    if end_year and end_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] <= int(end_year)]
    
    # Prepare yearly data for anomaly detection
    yearly_data = filtered_data.groupby('Year')[parameter].mean().reset_index()
    
    # Detect anomalies using the Temperature column
    anomaly_data = ml_models.detect_anomalies(
        yearly_data,
        features=[parameter]
    )
    
    # Create visualization
    fig = visualizer.plot_anomalies(
        anomaly_data,
        x_col='Year',
        y_col=parameter,
        title=f'Anomaly Detection for {parameter}',
        save_path=None
    )
    
    # Convert plot to base64
    plot_image = plot_to_base64(fig)
    
    # Get anomaly data
    anomalies = anomaly_data[anomaly_data['is_anomaly']][['Year', parameter, 'is_anomaly']].to_dict(orient='records')
    
    return jsonify({
        'plot': plot_image,
        'anomalies': anomalies
    })

@app.route('/api/temperature-trends')
def temperature_trends():
    """API endpoint for temperature trend analysis."""
    start_year = request.args.get('start_year', None)
    end_year = request.args.get('end_year', None)
    
    # Get data using lazy loading
    current_data = get_data()
    
    # Apply year filter if specified
    filtered_data = current_data
    if start_year and start_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] >= int(start_year)]
    if end_year and end_year.isdigit():
        filtered_data = filtered_data[filtered_data['Year'] <= int(end_year)]
    
    # Calculate yearly averages
    yearly_data = filtered_data.groupby('Year')['Temperature'].mean().reset_index()
    
    # Calculate rolling averages if not already in the data
    if 'temp_5yr_avg' not in yearly_data.columns:
        yearly_data['temp_5yr_avg'] = yearly_data['Temperature'].rolling(window=5, min_periods=1).mean()
    
    if 'temp_10yr_avg' not in yearly_data.columns:
        yearly_data['temp_10yr_avg'] = yearly_data['Temperature'].rolling(window=10, min_periods=1).mean()
    
    # Create multi-line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot yearly values
    ax.plot(yearly_data['Year'], yearly_data['Temperature'], 
            label='Annual Average', alpha=0.7, marker='o', markersize=3)
    
    # Plot 5-year moving average
    ax.plot(yearly_data['Year'], yearly_data['temp_5yr_avg'], 
            label='5-year Moving Average', linewidth=2)
    
    # Plot 10-year moving average
    ax.plot(yearly_data['Year'], yearly_data['temp_10yr_avg'], 
            label='10-year Moving Average', linewidth=3)
    
    # Add trend line
    x = yearly_data['Year']
    y = yearly_data['Temperature']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, label=f'Trend (Slope: {z[0]:.4f}°C/year)')
    
    title = 'Temperature Trends Over Time'
    if start_year and end_year:
        title += f' ({start_year}-{end_year})'
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64
    plot_image = plot_to_base64(fig)
    
    # Calculate trend statistics
    temp_change = yearly_data['Temperature'].iloc[-1] - yearly_data['Temperature'].iloc[0]
    total_years = yearly_data['Year'].iloc[-1] - yearly_data['Year'].iloc[0]
    avg_change_per_decade = (temp_change / total_years) * 10
    
    return jsonify({
        'plot': plot_image,
        'data': yearly_data.to_dict(orient='records'),
        'stats': {
            'total_change': f'{temp_change:.2f}°C',
            'change_per_decade': f'{avg_change_per_decade:.2f}°C',
            'trend_slope': f'{z[0]:.4f}°C/year',
            'total_years': int(total_years)
        }
    })

@app.route('/initialize')
def initialize():
    """Initialize the application data."""
    load_data()
    return jsonify({'status': 'Application initialized successfully'})

# Initialize data when the app starts using lazy loading approach
with app.app_context():
    try:
        print("Application starting. Data will be loaded on first request.")
    except Exception as e:
        print(f"Error during app initialization: {e}")

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    app.run(debug=True)