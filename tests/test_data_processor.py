from src.data_processor import DataProcessor
import os

def test_load_and_clean_data():
    processor = DataProcessor('data/climate_data.csv')
    df = processor.load_data()
    assert not df.empty
    df = processor.clean_data()
    assert df.isnull().sum().sum() == 0
