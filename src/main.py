from src.data_processor import DataProcessor
from src.algorithms import CustomTemperaturePredictor
from src.visualizer import plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_pipeline(file_path: str):
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.get_features_and_target()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CustomTemperaturePredictor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    plot_predictions(y_test, y_pred)
