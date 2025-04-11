from src.data_processor import DataProcessor
from src.algorithms import CustomTemperaturePredictor
from src.visualizer import plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_pipeline(file_path: str):
    """
    Primary runtime, runs all the methods required to load, process, and display the climate data

    Args:
        file_path (str): and string representation of the file that has the CSV data
    """
    # create the 'DataProcessor' object with the given 'file_path'
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.get_features_and_target()

    # train processed data with the machine learning model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CustomTemperaturePredictor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    # plot fully processed temperature and anomaly data on a graph
    plot_predictions(y_test, y_pred)
