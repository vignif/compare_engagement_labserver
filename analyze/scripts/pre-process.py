import os
import pandas as pd
import argparse
from pathlib import Path

class CSVMerger:
    def __init__(self, input_directory, output_file):
        self.input_directory = input_directory
        self.output_file = output_file

    def get_csv_files(self):
        """Get a list of all CSV files in the input directory."""
        return [f for f in os.listdir(self.input_directory) if f.endswith('.csv')]

    def merge_csv_files(self):
        """Merge all CSV files into a single CSV file."""
        csv_files = self.get_csv_files()
        merged_df = pd.DataFrame()

        for i, file in enumerate(csv_files):
            file_path = os.path.join(self.input_directory, file)
            df = pd.read_csv(file_path, sep=';')
            if i == 0:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df])

        merged_df.to_csv(self.output_file, index=False, sep=';')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge CSVs.')
    parser.add_argument('input_directory', type=str, help='input folder')
    parser.add_argument('file_output', type=str, help='output file')
    
    args = parser.parse_args()
    
    input_directory = Path(args.input_directory)
    file_output = Path(args.file_output)

    csv_merger = CSVMerger(input_directory, file_output)
    csv_merger.merge_csv_files()
    print(f"All CSV files have been merged into {file_output}")
