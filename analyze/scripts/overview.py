import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import argparse
from pathlib import Path

class DataOverview:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = None
        self.get_dataframe()
        
    def get_dataframe(self):
        """Load the DataFrame from the CSV file."""
        try:
            self.dataframe = pd.read_csv(self.file_path, sep=';')
        except Exception as e:
            raise ValueError(f"Error loading DataFrame: {e}")
    
    def calculate_statistics(self):
        """Calculate and return relevant statistics for each column."""
        if self.dataframe is None:
            raise ValueError("DataFrame is not loaded. Call get_dataframe() first.")
        
        # Initialize a dictionary to store statistics
        stats = {
            "Column": [],
            "Mean": [],
            "Median": [],
            "Standard Deviation": [],
            "Minimum": [],
            "Maximum": [],
            "Missing Values": [],
            "Zero Count": [],
            "Skewness": [],
            "Kurtosis": [],
            "Unique Values": [],
            "Top Value": [],
            "Top Frequency": []
        }

        # Iterate over each column
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype in [np.float64, np.int64]:
                stats["Column"].append(column)
                stats["Mean"].append(self.dataframe[column].mean())
                stats["Median"].append(self.dataframe[column].median())
                stats["Standard Deviation"].append(self.dataframe[column].std())
                stats["Minimum"].append(self.dataframe[column].min())
                stats["Maximum"].append(self.dataframe[column].max())
                stats["Missing Values"].append(self.dataframe[column].isnull().sum())
                stats["Zero Count"].append((self.dataframe[column] == 0).sum())
                stats["Skewness"].append(skew(self.dataframe[column].dropna()))
                stats["Kurtosis"].append(kurtosis(self.dataframe[column].dropna()))
                stats["Unique Values"].append(self.dataframe[column].nunique())
                stats["Top Value"].append(None)
                stats["Top Frequency"].append(None)
            else:
                # For non-numeric columns
                stats["Column"].append(column)
                stats["Mean"].append(None)
                stats["Median"].append(None)
                stats["Standard Deviation"].append(None)
                stats["Minimum"].append(None)
                stats["Maximum"].append(None)
                stats["Missing Values"].append(self.dataframe[column].isnull().sum())
                stats["Zero Count"].append(None)
                stats["Skewness"].append(None)
                stats["Kurtosis"].append(None)
                stats["Unique Values"].append(self.dataframe[column].nunique())
                top_value = self.dataframe[column].mode()
                stats["Top Value"].append(top_value[0] if not top_value.empty else None)
                stats["Top Frequency"].append(self.dataframe[column].value_counts().max())

        # Convert the dictionary to a DataFrame
        stats_df = pd.DataFrame(stats)
        
        return stats_df

    def get_additional_info(self):
        """Get additional information about the dataset."""
        num_rows = self.dataframe.shape[0]
        num_columns = self.dataframe.shape[1]
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_columns}")


    def count_value(self, column_name, value):
        """Count the occurrences of a specific value in a given column."""
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column {column_name} not found in DataFrame.")
        
        count = (self.dataframe[column_name] == value).sum()
        print(f"The value {value} appears {count} times in the column {column_name}.")
        return count
    
    def print_statistics(self):
        """Print the statistics in a readable format."""
        stats_df = self.calculate_statistics()
        print(stats_df.to_string(index=False))
        self.get_additional_info()
        print(self.dataframe.columns)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Overview')
    parser.add_argument('input_file', type=str, help='input file')
    
    args = parser.parse_args()
    
    file_path = Path(args.input_file)



    # file_path = 'csv/dataset_optimals.csv'  # Replace with the path to your CSV file

    overview = DataOverview(file_path)
    overview.print_statistics()
    overview.count_value('grace_eng', float(-1.0))

