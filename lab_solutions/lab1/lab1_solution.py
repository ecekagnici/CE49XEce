import pandas as pd
import os

def main():
    """Load the traffic dataset using an absolute path and analyze vehicle counts."""
    try:
        # Get the absolute path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset
        dataset_path = os.path.join(script_dir, "..", "..", "datasets", "traffic_data.csv")

        # Load dataset
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: The dataset file was not found. Please ensure 'traffic_data.csv' is located in the '/datasets/' folder.")
        print(f"Expected path: {dataset_path}")
        return

    # Compute basic descriptive statistics for 'vehicle_count'
    min_value = df['vehicle_count'].min()
    max_value = df['vehicle_count'].max()
    mean_value = df['vehicle_count'].mean()

    # Print the results
    print("Traffic Data Analysis:")
    print(f"Minimum vehicle count: {min_value}")
    print(f"Maximum vehicle count: {max_value}")
    print(f"Mean vehicle count: {mean_value:.2f}")

if __name__ == '__main__':
    main()
