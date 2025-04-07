import argparse
from src.main import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Climate Change Impact Analyzer")
    parser.add_argument('--data', type=str, required=True, help='Path to climate data CSV')
    args = parser.parse_args()
    run_pipeline(args.data)

if __name__ == "__main__":
    main()
