Climate Change Impact Analyzer
A web application that visualizes and analyzes climate change trends using historical temperature data. This project allows users to interactively explore temperature patterns, seasonal variations, decade comparisons, future predictions, and anomaly detection.
Features

Temperature Trends Visualization: View monthly, yearly, and seasonal temperature trends over time
Seasonal Analysis: Analyze how temperature patterns vary across seasons for different decades
Decade Comparison: Compare average temperatures across different decades
Future Predictions: Generate temperature forecasts using machine learning models
Anomaly Detection: Identify unusual temperature patterns and outliers in the data

Setup

Clone the repository
git clone https://github.com/yourusername/ClimateChangeAnalysis.git
cd ClimateChangeAnalysis

Create a virtual environment
python -m venv venv

Activate the virtual environment

Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate


Install dependencies
pip install -r requirements.txt

Ensure the data directory structure exists
mkdir -p data/raw

Add climate data

Place your climate data CSV file at data/raw/climate_data.csv
Or use the fallback data generator built into the application



Usage
Run the Web Application
cd src
python web_app.py
Then open your browser and navigate to: http://localhost:5000
Interactive Features

Year Range Selection: Filter data by selecting start and end years
Time Scale Options: View data in monthly, yearly, or seasonal format
Decade Selection: Compare specific decades or view all decades together
Future Prediction: Adjust the number of years to predict into the future

Project Structure

src/: Source code

web_app.py: Flask web application
simple_climate_data_processor.py: Data processing utilities
ml_models.py: Machine learning models for prediction and anomaly detection
visualization.py: Data visualization functions
templates/: HTML templates for the web interface


data/: Climate data files

raw/: Raw data files

climate_data.csv: Primary climate data file





Data Format
The application expects a CSV file with the following columns:

Year: Numeric year value
Month: Numeric month value (1-12)
Temperature: Temperature value (typically anomaly in °C)

Dependencies

Flask: Web framework
Pandas: Data manipulation
NumPy: Numerical computing
Matplotlib: Data visualization
Seaborn: Advanced visualization
Scikit-learn: Machine learning algorithms

Contributing

Fork the repository
Create a feature branch
Make your changes
Submit a pull request

Credits

Temperature data source: Berkeley Earth Surface Temperature Study
CIS 4930, Spring 2025
Group Members:
Dudleen Paul
Kensia Saint-Hilaire
Bruno Page
