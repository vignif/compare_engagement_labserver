import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

class DataAnalyzer:
    def __init__(self, file_path, c1, c2):
        self.file_path = file_path
        self.dataframe = self.load_dataframe()
        self.c1 = c1
        self.c2 = c2

    def load_dataframe(self):
        """Load the DataFrame from the CSV file."""
        try:
            dataframe = pd.read_csv(self.file_path, sep=';')
            return dataframe
        except Exception as e:
            raise ValueError(f"Error loading DataFrame: {e}")

    def calculate_mse_per_instance(self):
        """Calculate MSE between two columns for each instance (row)."""
        mse_values = []
        for index, row in self.dataframe.iterrows():
            mse = (row[self.c1] - row[self.c2]) ** 2
            mse_values.append(mse)
        return mse_values

    def mse(self):
        mse = ((self.dataframe[self.c1] - self.dataframe[self.c2]) ** 2).mean()
        return mse
    
    def correlation(self):
        """Calculate and print the correlation between two columns."""
        correlation = self.dataframe[self.c1].corr(self.dataframe[self.c2])
        return correlation
    
    def plot_mse(self, mse_values):
        """Plot the MSE values."""
        plt.figure(figsize=(10, 6))
        plt.plot(mse_values, marker='o', linestyle='-', color='b')
        plt.title('MSE per Instance')
        plt.xlabel('Instance')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.savefig(f"mse_{self.file_path.name}.png")

    def plot_metrics(self):
        """Plot two metrics on the same figure."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.dataframe[self.c1], label=self.c1, marker='o', linestyle='-')
        plt.plot(self.dataframe[self.c2], label=self.c2, marker='x', linestyle='-')
        plt.title(f'{self.c1} and {self.c2} Comparison')
        plt.xlabel('Instance')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"compare_{self.file_path.name}.png")

    def analyze_and_plot(self):
        """Calculate and plot MSE for the given columns, and plot the metrics."""
        mse_values = self.calculate_mse_per_instance()
        self.plot_mse(mse_values)
        self.plot_metrics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and plot metrics from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    column1 = "grace_eng"
    column2 = "eng_1"

    analyzer = DataAnalyzer(file_path, column1, column2)
    analyzer.analyze_and_plot()
    
    print(f"Overall MSE: {analyzer.mse():.4f}")
    print(f"Correlation: {analyzer.correlation():.4f}")
