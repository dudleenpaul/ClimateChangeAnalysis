### CIS 4930, Spring 2025
Group Members:
Dudleen Paul
Kensia Saint-Hilaire
Bruno Page

# Climate Change Impact Analyzer
A web application that visualizes and analyzes climate change trends using historical temperature data. This project allows users to interactively explore temperature patterns, seasonal variations, decade comparisons, future predictions, and anomaly detection.

## Features
*Temperature Trends Visualization:* View monthly, yearly, and seasonal temperature trends over time\
*Seasonal Analysis:* Analyze how temperature patterns vary across seasons for different decades\
*Decade Comparison:* Compare average temperatures across different decades\
*Future Predictions:* Generate temperature forecasts using machine learning models\
*Anomaly Detection:* Identify unusual temperature patterns and outliers in the data\

## Setup

Clone the repository
```
git clone https://github.com/yourusername/ClimateChangeAnalysis.git
cd ClimateChangeAnalysis
```

Create a virtual environment
```python -m venv venv```

Activate the virtual environment
Windows: ```venv\Scripts\activate```\
macOS/Linux: ```source venv/bin/activate```\

Install dependencies
```pip install -r requirements.txt```\

Ensure the data directory structure exists
```mkdir -p data/raw```\

## Usage
Run the Web Application
make sure you are in the ClimateChangeAnalysis folder
run "python3 main.py web" in the command line
Then open your browser and navigate to: http://localhost:5000


## Interactive Features
**Year Range Selection:** Filter data by selecting start and end years
**Time Scale Options:** View data in monthly, yearly, or seasonal format
**Decade Selection:** Compare specific decades or view all decades together
**Future Prediction:** Adjust the number of years to predict into the future


## Project Structure
```
ClimateChangeAnalysis/
│
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   │   └── climate_data.csv  # Primary climate data file
│   └── output/               # Output files from analysis
│
├── src/                      # Source code
│   ├── templates/            # Flask HTML templates
│   │   └── index.html        # Main dashboard template
│   ├── __init__.py           # Package initialization
│   ├── algorithms.py         # Core algorithms for data analysis
│   ├── anomaly_detector.py   # Anomaly detection algorithms
│   ├── cli.py                # Command-line interface
│   ├── climate_cluster.py    # Clustering algorithms for climate data
│   ├── data_fetcher.py       # Utilities for fetching external data
│   ├── data_processor.py     # Data processing and transformation utilities
│   ├── ml_models.py          # Machine learning models implementation
│   ├── simple_climate_data_processor.py  # Simplified data processing
│   ├── visualization.py      # Data visualization functions
│   └── web_app.py            # Flask web application
│
├── tests/                    # Test directory
│   ├── __pycache__/          # Python cache files
│   ├── .pytest_cache/        # Pytest cache
│   ├── visualizations/       # Test visualization outputs
│   ├── __init__.py           # Test package initialization
│   ├── mock_climate_data.csv # Mock data for testing
│   ├── run_tests.py          # Script to run all tests
│   ├── test_algorithms.py    # Tests for algorithms
│   ├── test_data_processor.py # Tests for data processor
│   ├── test_integration.py   # Integration tests
│   ├── test_ml_models.py     # Tests for ML models
│   ├── test_utils.py         # Tests for utility functions
│   ├── test_visualization.py # Tests for visualization functions
│   └── test_web_app.py       # Tests for web application
│
├── .coverage                 # Coverage reports
├── .gitignore                # Git ignore file
├── main.py                   # Main entry point script
├── README.md                 # Project documentation
└── requirements.txt          # Project dependencies
```

## Data Format
The application expects a CSV file with the following columns:

**Year:** Numeric year value
**Month:** Numeric month value (1-12)
**Temperature:** Temperature value (typically anomaly in °C)

## Dependencies

**Flask:** Web framework
**Pandas:** Data manipulation
**NumPy:** Numerical computing
**Matplotlib:** Data visualization
**Seaborn:** Advanced visualization
**Scikit-learn:** Machine learning algorithms

**Credits**
Temperature data source: Berkeley Earth Surface Temperature Study
