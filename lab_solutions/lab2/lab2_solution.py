import pandas as pd
import os

def load_data():
    """Load dataset using an absolute path."""
    try:
        # Get the absolute path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset
        dataset_path = os.path.join(script_dir, "..", "..", "datasets", "soil_test.csv")

        data = pd.read_csv(dataset_path)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}")
        return None

def clean_data(file, column):
    """Fill missing values and handle outliers using loops."""
    if file is None:
        return
    
    n = len(file[column])
    avg = file[column].mean()

    # Handling missing values with the column average
    for i in range(n):
        if pd.isna(file.loc[i, column]):
            file.loc[i, column] = avg

    # Compute standard deviation
    standard_dev = file[column].std()
    min_threshold = avg - standard_dev * 1.5
    max_threshold = avg + standard_dev * 1.5

    # Handling outliers using standard deviation
    for i in range(n):
        if not (min_threshold <= file.loc[i, column] <= max_threshold):
            file.loc[i, column] = avg

def compute_statistics(file, column):
    """Print basic statistics of the dataset."""
    if file is None:
        return
    
    print("\nSoil Data Analysis:")
    print(f"Minimum {column} value: {file[column].min()}")
    print(f"Maximum {column} value: {file[column].max()}")
    print(f"Mean {column} value: {file[column].mean():.2f}")
    print(f"Median {column} value: {file[column].median()}")
    print(f"Standard Deviation {column} value: {file[column].std():.2f}")

def main():
    processed_column = "nitrogen"  # Column to analyze
    data = load_data()
    if data is not None:
        clean_data(data, processed_column)
        compute_statistics(data, processed_column)

if __name__ == "__main__":
    main()
