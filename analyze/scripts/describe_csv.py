import pandas as pd
import argparse
import os

def load_csv(file_path):
    """Loads a CSV file into a pandas DataFrame with chunking."""
    try:
        # Read the file in chunks to avoid memory overload on large datasets
        chunk_size = 100000  # Adjust the chunk size based on your system's capacity
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
        return pd.concat(chunk_iter, ignore_index=True)  # Combine chunks into one DataFrame
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or improperly formatted.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the file: {e}")

def display_overview(df):
    """Displays an overview of the DataFrame."""
    print("\n--- Dataset Overview ---")
    print("\nFirst 5 Rows:")
    print(df.head())
    
    # Get the basic information about the dataset
    print("\nDataset Info (optimized):")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Column Names: {', '.join(df.columns)}")
    print("\nSummary Statistics:")
    print(df.describe(include="all").transpose())

def analyze_missing_values(df):
    """Analyzes and displays missing values in the DataFrame."""
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0])

def analyze_unique_values(df):
    """Analyzes and displays unique value counts for each column."""
    print("\n--- Unique Values per Column ---")
    unique_counts = {col: df[col].nunique() for col in df.columns}
    for col, count in unique_counts.items():
        print(f"{col}: {count} unique values")

def main(file_path):
    """Main function to analyze the provided CSV file."""
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Load and analyze the dataset
    try:
        df = load_csv(file_path)
        display_overview(df)
        analyze_missing_values(df)
        analyze_unique_values(df)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a CSV file and provide an overview of its contents.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    main(args.file_path)
