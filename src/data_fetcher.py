# src/data_fetcher.py
import requests
import os
import pandas as pd
import io

def download_noaa_global_temp(save_path: str = "data/climate_data.csv") -> None:
    """
    Downloads monthly global land temperature data from NOAA via DataHub.
    Cleans and saves it as a ready-to-process CSV.

    Returns without saving data if we fail to download the data

    Kwargs:
        save_path (str): a string representation of the filepath to save the climate data to
    Returns:
        None
    """
    url = "https://datahub.io/core/global-temp/r/monthly.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[INFO] Downloading climate data from {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"[ERROR] Failed to download data. Status code: {response.status_code}")
        return

    df = pd.read_csv(io.StringIO(response.text))
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df = df.rename(columns={'Mean': 'temperature'})
    df = df[['year', 'month', 'temperature']]  # simplify to needed columns

    df.to_csv(save_path, index=False)
    print(f"[✓] Climate data saved to {save_path}")
